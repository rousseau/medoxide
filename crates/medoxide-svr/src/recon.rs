//! Reconstruction sur grille (étape 4a) : la grille du volume à estimer et son initialisation par **adjoint normalisé**.
//!
//! Le volume reconstruit est une grille isotrope **alignée sur les axes du monde** (RAS+), qui ne dépend d'aucun stack. Son
//! étendue est la boîte englobante de tous les voxels de masque cérébral de tous les stacks, plus une marge.

use ndarray::{Array2, Array3, Axis};
use nalgebra::{Matrix4, Vector3};

use crate::{Stack, SvrError, Volume};

/// Support minimal (somme des poids rétroprojetés) au-dessous duquel un voxel est considéré comme non couvert par les données.
/// Chaque pixel de masque répartit une masse totale de 1 sur les voxels : le support compte, en pixels, les données qui touchent
/// le voxel. 1e-3 écarte les voxels que seule la queue d'une PSF atteint.
pub const SUPPORT_MIN: f32 = 1e-3;

/// Géométrie d'une grille de reconstruction : affine voxel → monde (axes du monde, voxels isotropes) et dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    /// Affine voxel → monde : `diag(résolution)` et translation = position monde du centre du voxel `(0, 0, 0)`.
    pub affine: Matrix4<f64>,
    /// Nombre de voxels par axe.
    pub dims: (usize, usize, usize),
    /// Taille du voxel isotrope, en mm.
    pub resolution_mm: f64,
}

impl GridSpec {
    /// Un [`Volume`] nul de cette géométrie.
    pub fn zeros(&self) -> Volume {
        Volume::new(Array3::<f32>::zeros(self.dims), self.affine).expect("l'affine d'une grille est diagonale, de pas > 0 : inversible")
    }
}

/// **Grille de reconstruction** : boîte englobante, élargie de `margin_mm` de chaque côté, des positions monde de tous les voxels
/// de masque (nettoyés, voir [`crate::BrainMask`]) de tous les `stacks`, en voxels isotropes de `resolution_mm` alignés sur les
/// axes du monde. Le coin minimal de la boîte est le centre du voxel `(0, 0, 0)` ; `n = ⌈(haut − bas) / résolution⌉ + 1` par axe.
///
/// # Erreurs
/// [`SvrError::NoStacks`] ; [`SvrError::InvalidGrid`] (résolution ≤ 0 ou marge < 0, non finies) ;
/// [`SvrError::NoMask`] si un stack n'a pas de masque attaché.
pub fn reconstruction_grid(stacks: &[Stack], resolution_mm: f64, margin_mm: f64) -> Result<GridSpec, SvrError> {
    if stacks.is_empty() {
        return Err(SvrError::NoStacks);
    }
    // `!(x > 0.0)` est vrai aussi pour NaN.
    if !(resolution_mm > 0.0) || !resolution_mm.is_finite() || !(margin_mm >= 0.0) || !margin_mm.is_finite() {
        return Err(SvrError::InvalidGrid);
    }
    let mut bas = Vector3::from_element(f64::INFINITY);
    let mut haut = Vector3::from_element(f64::NEG_INFINITY);
    for stack in stacks {
        let masque = stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
        let a = stack.affine();
        for ((i, j, k), &dedans) in masque.voxels().indexed_iter() {
            if dedans {
                let monde = (a * nalgebra::Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
                bas = bas.inf(&monde);
                haut = haut.sup(&monde);
            }
        }
    }
    let marge = Vector3::from_element(margin_mm);
    let (bas, haut) = (bas - marge, haut + marge);
    let n = |a: usize| ((haut[a] - bas[a]) / resolution_mm).ceil() as usize + 1;
    let mut affine = Matrix4::identity();
    for a in 0..3 {
        affine[(a, a)] = resolution_mm;
        affine[(a, 3)] = bas[a];
    }
    Ok(GridSpec { affine, dims: (n(0), n(1), n(2)), resolution_mm })
}

/// Résultat de [`normalized_adjoint`].
#[derive(Debug, Clone)]
pub struct NormalizedAdjoint {
    /// Volume initial `x₀` : moyenne pondérée des pixels qui touchent chaque voxel ; 0 là où le support est inférieur à [`SUPPORT_MIN`].
    pub image: Array3<f32>,
    /// Support : somme des poids rétroprojetés (compte, en pixels de masque, des données qui touchent le voxel).
    pub support: Array3<f32>,
}

/// **Adjoint normalisé** : `x₀ = Σ Aₖᵀ (mₖ ⊙ yₖ) / Σ Aₖᵀ mₖ`, où `yₖ` sont les pixels de la coupe `k`, `mₖ` son masque cérébral (0 ou 1)
/// et `Aₖᵀ` la transposée **exacte** de l'opérateur d'acquisition ([`Volume::back_project`]). C'est, en chaque voxel, la moyenne des
/// pixels de masque qui le touchent, pondérée par les coefficients de l'opérateur.
///
/// Toutes les coupes de tous les `stacks` (masque requis) sont utilisées, avec leur [`crate::Slice::psf`]. Les intensités sont prises
/// telles quelles : leur mise à l'échelle entre stacks relève de l'étape 4c.
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque attaché.
pub fn normalized_adjoint(grid: &GridSpec, stacks: &[Stack]) -> Result<NormalizedAdjoint, SvrError> {
    let geometrie = grid.zeros();
    let mut numerateur = Array3::<f64>::zeros(grid.dims);
    let mut denominateur = Array3::<f64>::zeros(grid.dims);
    for stack in stacks {
        let masque = stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
        for coupe in stack.slices() {
            let m: Array2<f32> = masque.voxels().index_axis(Axis(2), coupe.index()).mapv(|v| f32::from(u8::from(v)));
            let y: Array2<f32> = &coupe.data() * &m;
            let psf = coupe.psf();
            geometrie.back_project(&coupe, &psf, y.view(), &mut numerateur);
            geometrie.back_project(&coupe, &psf, m.view(), &mut denominateur);
        }
    }
    let mut image = Array3::<f32>::zeros(grid.dims);
    let mut support = Array3::<f32>::zeros(grid.dims);
    for ((idx, &d), s) in denominateur.indexed_iter().zip(support.iter_mut()) {
        *s = d as f32;
        if *s >= SUPPORT_MIN {
            image[idx] = (numerateur[idx] / d) as f32;
        }
    }
    Ok(NormalizedAdjoint { image, support })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector4};
    use nifti::{writer::WriterOptions, NiftiHeader};
    use std::path::{Path, PathBuf};

    fn racine() -> String {
        format!("{}/../..", env!("CARGO_MANIFEST_DIR"))
    }

    /// Dossier temporaire propre à un test (nom + processus), supprimé en fin de test.
    struct Temp(PathBuf);
    impl Temp {
        fn new(nom: &str) -> Temp {
            let d = std::env::temp_dir().join(format!("medoxide_recon_{nom}_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Temp(d)
        }
        fn fichier(&self, nom: &str) -> PathBuf {
            self.0.join(nom)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn en_tete(affine: &Matrix4<f64>, pixdim: [f32; 3]) -> NiftiHeader {
        let mut h = NiftiHeader::default();
        h.sform_code = 1;
        h.srow_x = std::array::from_fn(|c| affine[(0, c)] as f32);
        h.srow_y = std::array::from_fn(|c| affine[(1, c)] as f32);
        h.srow_z = std::array::from_fn(|c| affine[(2, c)] as f32);
        h.pixdim = [1.0, pixdim[0], pixdim[1], pixdim[2], 1.0, 1.0, 1.0, 1.0];
        h
    }

    /// Écrit un stack (image et masque) d'affine donnée et le relit, masque attaché.
    fn stack_avec_masque(t: &Temp, nom: &str, data: &Array3<f32>, masque: &Array3<u8>, affine: &Matrix4<f64>, pixdim: [f32; 3]) -> Stack {
        let h = en_tete(affine, pixdim);
        let (image, m) = (t.fichier(&format!("{nom}.nii.gz")), t.fichier(&format!("{nom}_mask.nii.gz")));
        WriterOptions::new(&image).reference_header(&h).write_nifti(data).unwrap();
        WriterOptions::new(&m).reference_header(&h).write_nifti(masque).unwrap();
        let mut stack = Stack::read(&image).unwrap();
        stack.set_brain_mask(&m).unwrap();
        stack
    }

    /// Affine `R · diag(échelle)` + origine telle que l'indice `centre` tombe en `cible`.
    fn affine_centree(r: &Rotation3<f64>, echelle: [f64; 3], centre: [f64; 3], cible: [f64; 3]) -> Matrix4<f64> {
        let lineaire = r.matrix() * nalgebra::Matrix3::from_diagonal(&Vector3::from(echelle));
        let origine = Vector3::from(cible) - lineaire * Vector3::from(centre);
        let mut a = Matrix4::identity();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&lineaire);
        a.fixed_view_mut::<3, 1>(0, 3).copy_from(&origine);
        a
    }

    /// Critère 1 : grille identique à la référence indépendante (nibabel et scipy) sur les trois stacks TRUFI de 3 sujets réels,
    /// pour 3 couples résolution / marge.
    #[test]
    fn grid_matches_the_python_reference_on_real_stacks() {
        let tsv = std::fs::read_to_string(format!("{}/data/reference/grid/grilles.tsv", racine())).expect("références absentes : lancer scripts/make_reference_grid.py");
        let mut n_lignes = 0;
        for ligne in tsv.lines() {
            let c: Vec<&str> = ligne.split('\t').collect();
            let (resolution, marge): (f64, f64) = (c[0].parse().unwrap(), c[1].parse().unwrap());
            let bas: [f64; 3] = [c[3].parse().unwrap(), c[4].parse().unwrap(), c[5].parse().unwrap()];
            let n: [usize; 3] = [c[6].parse().unwrap(), c[7].parse().unwrap(), c[8].parse().unwrap()];
            let stacks: Vec<Stack> = c[9]
                .split(',')
                .map(|chemin| {
                    let mut s = Stack::read(Path::new(&format!("{}/{chemin}", racine()))).unwrap();
                    let (dossier, nom) = chemin.rsplit_once("/anat/").unwrap();
                    let (sujet_ses, _) = (dossier.strip_prefix("data/svr/jeu_reel_tru_haste/").unwrap(), ());
                    let masque = format!("{}/data/svr/jeu_reel_tru_haste/derivatives/medx-fetalbet/{sujet_ses}/anat/{}", racine(), nom.replace("_T2w.nii.gz", "_desc-brain_mask.nii.gz"));
                    s.set_brain_mask(Path::new(&masque)).unwrap();
                    s
                })
                .collect();
            let g = reconstruction_grid(&stacks, resolution, marge).unwrap();
            assert_eq!((g.dims.0, g.dims.1, g.dims.2), (n[0], n[1], n[2]), "{} résolution {resolution} marge {marge}", c[2]);
            for a in 0..3 {
                assert!((g.affine[(a, 3)] - bas[a]).abs() < 1e-6, "{} axe {a} : {} contre {}", c[2], g.affine[(a, 3)], bas[a]);
                assert_eq!(g.affine[(a, a)], resolution);
            }
            n_lignes += 1;
        }
        assert_eq!(n_lignes, 9);
    }

    /// Géométrie calculée à la main : trois voxels de masque connexes (diagonale) d'un stack d'affine diagonale.
    #[test]
    fn grid_covers_the_mask_with_the_margin() {
        let t = Temp::new("grille");
        let affine = Matrix4::new(1.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 20.0, 0.0, 0.0, 3.0, 30.0, 0.0, 0.0, 0.0, 1.0);
        let mut masque = Array3::<u8>::zeros((6, 5, 4));
        for k in 0..3 {
            masque[[k, k, k]] = 1; // positions monde (10, 20, 30), (11, 22, 33), (12, 24, 36)
        }
        let stack = stack_avec_masque(&t, "s", &Array3::from_elem((6, 5, 4), 1.0), &masque, &affine, [1.0, 2.0, 3.0]);
        let g = reconstruction_grid(&[stack], 1.0, 2.0).unwrap();
        // bas = (8, 18, 28), haut = (14, 26, 38) -> n = (6+1, 8+1, 10+1)
        assert_eq!((g.dims.0, g.dims.1, g.dims.2), (7, 9, 11));
        assert_eq!([g.affine[(0, 3)], g.affine[(1, 3)], g.affine[(2, 3)]], [8.0, 18.0, 28.0]);
        assert!(g.affine.fixed_view::<3, 3>(0, 0).determinant() > 0.0);
        let v = g.zeros();
        assert_eq!(v.dim(), g.dims);
        // le voxel du coin haut contient (14, 26, 38) à l'arrondi près : le dernier centre est à bas + (n-1)·res >= haut
        assert!(g.affine[(0, 3)] + 6.0 >= 14.0 && g.affine[(2, 3)] + 10.0 >= 38.0);
    }

    #[test]
    fn grid_rejects_invalid_inputs() {
        let t = Temp::new("grille_erreurs");
        let affine = Matrix4::identity();
        let masque = Array3::<u8>::from_elem((3, 3, 3), 1);
        let avec = stack_avec_masque(&t, "a", &Array3::from_elem((3, 3, 3), 1.0), &masque, &affine, [1.0, 1.0, 1.0]);
        assert!(matches!(reconstruction_grid(&[], 1.0, 0.0), Err(SvrError::NoStacks)));
        for (res, marge) in [(0.0, 1.0), (-1.0, 1.0), (f64::NAN, 1.0), (f64::INFINITY, 1.0), (1.0, -0.1), (1.0, f64::NAN)] {
            assert!(matches!(reconstruction_grid(std::slice::from_ref(&avec), res, marge), Err(SvrError::InvalidGrid)), "{res} / {marge}");
        }
        let sans = Stack::read(&t.fichier("a.nii.gz")).unwrap();
        assert!(matches!(reconstruction_grid(&[sans], 1.0, 0.0), Err(SvrError::NoMask(_))));
    }

    /// Critère 2 : si toutes les coupes viennent d'un volume constant, l'adjoint normalisé redonne la constante partout où le support
    /// est positif, même avec un masque partiel (il entre dans le numérateur et le dénominateur).
    #[test]
    fn normalized_adjoint_of_constant_slices_is_constant() {
        let t = Temp::new("constante");
        let mut alea = 0x9E37_79B9_7F4A_7C15_u64;
        let mut suivant = move || {
            alea ^= alea << 13;
            alea ^= alea >> 7;
            alea ^= alea << 17;
            (alea >> 11) as f64 / (1u64 << 53) as f64
        };
        let axial = affine_centree(&Rotation3::from_euler_angles(0.1, -0.15, 0.2), [1.0, 1.0, 2.0], [11.5, 11.5, 1.5], [0.0, 0.0, 0.0]);
        let coronal = affine_centree(&Rotation3::from_euler_angles(std::f64::consts::FRAC_PI_2 + 0.1, 0.05, -0.1), [1.0, 1.0, 2.0], [11.5, 11.5, 1.5], [1.0, 0.5, -0.5]);
        let stacks: Vec<Stack> = [("ax", axial), ("cor", coronal)]
            .iter()
            .map(|(nom, a)| {
                let masque = Array3::from_shape_fn((24, 24, 4), |_| u8::from(suivant() < 0.6)); // masque partiel : 60 % des pixels
                stack_avec_masque(&t, nom, &Array3::from_elem((24, 24, 4), 5.0), &masque, a, [1.0, 1.0, 2.0])
            })
            .collect();
        // BrainMask::read garde la plus grande composante connexe : le masque effectif est un sous-ensemble du tirage
        let grille = reconstruction_grid(&stacks, 1.0, 3.0).unwrap();
        let r = normalized_adjoint(&grille, &stacks).unwrap();
        let couverts = r.support.iter().filter(|&&s| s >= SUPPORT_MIN).count();
        let pire = r.image.iter().zip(r.support.iter()).filter(|(_, &s)| s >= SUPPORT_MIN).map(|(&v, _)| (v - 5.0).abs()).fold(0.0_f32, f32::max);
        println!("{couverts} voxels couverts sur {} ; écart max à la constante : {pire:.1e}", r.image.len());
        assert!(couverts > 1000, "trop peu de voxels couverts : {couverts}");
        assert!(pire < 1e-4, "{pire}");
        // hors support : 0
        assert!(r.image.iter().zip(r.support.iter()).filter(|(_, &s)| s < SUPPORT_MIN).all(|(&v, _)| v == 0.0));
    }
    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let (va, vb): (f64, f64) = (a.iter().map(|x| (x - ma).powi(2)).sum(), b.iter().map(|y| (y - mb).powi(2)).sum());
        cov / (va * vb).sqrt()
    }

    /// PSNR (dB) de `a` par rapport à la référence `b`, pour une dynamique `plage`.
    fn psnr(a: &[f64], b: &[f64], plage: f64) -> f64 {
        let eqm = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
        20.0 * (plage / eqm.sqrt()).log10()
    }

    /// Critère 3 (descriptif, ligne de base pour la suite) : trois stacks (axial, coronal, sagittal ; pixels 0,8 mm, épaisseur 3,5 mm,
    /// légèrement obliques) simulés par l'opérateur de l'étape 2 depuis l'atlas de Gholipour, sans mouvement ni bruit ; qualité de
    /// l'adjoint normalisé par rapport à l'atlas, comparée à chacun des stacks pris seul (trilinéaire). À lancer en
    /// `--release --ignored --nocapture` (≈ 1 min).
    #[test]
    #[ignore = "données locales ; lent en debug"]
    fn normalized_adjoint_on_stacks_simulated_from_the_atlas() {
        let t = Temp::new("atlas");
        let chemin = format!("{}/data/atlas/gholipour/STA21.nii.gz", racine());
        let atlas = Volume::from_stack(&Stack::read(Path::new(&chemin)).expect("atlas absent : voir data/atlas/gholipour"));
        let demi_tour = std::f64::consts::FRAC_PI_2;
        let specs = [
            ("axial", Rotation3::from_euler_angles(0.05, -0.04, 0.03)),
            ("coronal", Rotation3::from_euler_angles(demi_tour + 0.04, 0.03, -0.05)),
            ("sagittal", Rotation3::from_euler_angles(0.03, demi_tour - 0.04, 0.05)),
        ];
        let dims = (188, 188, 40);
        let stacks: Vec<Stack> = specs
            .iter()
            .map(|(nom, r)| {
                let affine = affine_centree(r, [0.8, 0.8, 3.5], [93.5, 93.5, 19.5], [0.0, 0.0, 0.0]);
                // une première lecture fournit la géométrie des coupes ; l'opérateur les simule ensuite depuis l'atlas
                let modele = stack_avec_masque(&t, &format!("{nom}_modele"), &Array3::zeros(dims), &Array3::from_elem(dims, 1u8), &affine, [0.8, 0.8, 3.5]);
                let (mut data, mut masque) = (Array3::<f32>::zeros(dims), Array3::<u8>::zeros(dims));
                for coupe in modele.slices() {
                    let sim = atlas.simulate_slice(&coupe, &coupe.psf());
                    for ((i, j), &v) in sim.values.indexed_iter() {
                        data[[i, j, coupe.index()]] = v;
                        masque[[i, j, coupe.index()]] = u8::from(sim.coverage[[i, j]] >= 0.99 && v > 150.0);
                    }
                }
                stack_avec_masque(&t, nom, &data, &masque, &affine, [0.8, 0.8, 3.5])
            })
            .collect();
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let debut = std::time::Instant::now();
        let r = normalized_adjoint(&grille, &stacks).unwrap();
        println!("grille {:?} ({} voxels) ; adjoint normalisé en {:.1} s", grille.dims, r.image.len(), debut.elapsed().as_secs_f64());
        let singles: Vec<Volume> = stacks.iter().map(Volume::from_stack).collect();
        // voxels de comparaison : support suffisant, tissu de l'atlas, et lisibles dans les trois stacks pour les lignes de base
        let (mut x0, mut verite) = (Vec::new(), Vec::new());
        let mut bases: Vec<Vec<f64>> = vec![Vec::new(); 3];
        for ((i, j, k), &s) in r.support.indexed_iter() {
            if s < 0.5 {
                continue;
            }
            let monde = (grille.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
            let a = match atlas.sample(&monde) {
                Some(a) if a > 150.0 => f64::from(a),
                _ => continue,
            };
            let valeurs: Vec<Option<f32>> = singles.iter().map(|v| v.sample(&monde)).collect();
            if valeurs.iter().any(Option::is_none) {
                continue;
            }
            x0.push(f64::from(r.image[[i, j, k]]));
            verite.push(a);
            for (b, v) in bases.iter_mut().zip(valeurs) {
                b.push(f64::from(v.unwrap()));
            }
        }
        let plage = {
            let mut v = verite.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[(v.len() as f64 * 0.99) as usize]
        };
        println!("{} voxels de comparaison ; dynamique (p99 de l'atlas) {plage:.0}", verite.len());
        println!("adjoint normalisé : NCC {:.4}, PSNR {:.2} dB", correlation(&x0, &verite), psnr(&x0, &verite, plage));
        for (nom, b) in specs.iter().map(|s| s.0).zip(&bases) {
            println!("stack {nom:9} seul (trilinéaire) : NCC {:.4}, PSNR {:.2} dB", correlation(b, &verite), psnr(b, &verite, plage));
        }
        assert!(verite.len() > 100_000, "trop peu de voxels de comparaison");
    }
}
