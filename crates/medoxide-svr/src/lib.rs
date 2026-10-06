//! `medoxide-svr` : reconstruction coupe-vers-volume (SVR) en IRM fœtale.
//!
//! Pour l'instant : la lecture d'un stack (volume 2D multi-coupes) avec sa géométrie.
//!
//! # Conventions
//!
//! - Repère monde : **RAS+ en mm**, celui du `sform` NIfTI (comme `nibabel`). ITK et
//!   SimpleITK travaillent en LPS (signes de x et y inversés) : à convertir aux frontières.
//! - Un indice de voxel `(i, j, k)` désigne le **centre** du voxel (indices entiers).
//! - Une coupe est un `k` fixé sur le **3ᵉ axe** du fichier.
//! - L'affine voxel → monde peut avoir un **déterminant négatif** (repère d'indices « main
//!   gauche » : c'est le cas courant, y compris des 96 stacks du jeu de développement) :
//!   cela n'est pas une erreur, et l'image n'est jamais retournée.

use std::path::{Path, PathBuf};

use medoxide_core::{read_volume, volume_info, CoreError};
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use ndarray::{Array3, ArrayView2, Axis};

/// Cosinus maximal toléré entre deux colonnes de l'affine : au-delà, les axes ne sont pas
/// orthogonaux (cisaillement) et la coupe n'a plus de repère rigide.
const COSINE_MAX: f64 = 1e-3;
/// Écart maximal toléré, en mm, entre `pixdim` et la norme d'une colonne de l'affine.
const SPACING_TOLERANCE_MM: f64 = 1e-3;

/// Erreurs de lecture d'un stack.
#[derive(Debug)]
pub enum SvrError {
    /// Lecture NIfTI impossible, ou fichier qui n'est pas un volume 3D ; contient le fichier fautif
    /// et l'erreur d'origine.
    Read { path: PathBuf, source: CoreError },
    /// L'en-tête n'a pas de `sform` (`sform_code == 0`) : pas d'affine exploitable.
    NoSform(PathBuf),
    /// Une colonne de l'affine est nulle ou non finie : géométrie dégénérée.
    DegenerateAffine(PathBuf),
    /// Deux axes ne sont pas orthogonaux ; contient le plus grand cosinus mesuré.
    ShearedAffine { path: PathBuf, cosine: f64 },
    /// Le masque cérébral n'a pas la même grille que le stack (dimensions ou affine) ; `detail` explique.
    MaskGridMismatch { path: PathBuf, detail: String },
    /// Le masque cérébral ne contient aucun voxel.
    EmptyMask(PathBuf),
    /// Le stack n'a pas de masque cérébral attaché (nécessaire pour le placer dans un groupe).
    NoMask(PathBuf),
    /// `pixdim` et les normes des colonnes de l'affine diffèrent de plus de 1e-3 mm.
    InconsistentSpacing {
        path: PathBuf,
        pixdim: [f64; 3],
        columns: [f64; 3],
    },
}

impl std::fmt::Display for SvrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvrError::Read { path, source } => write!(f, "{} : {source}", path.display()),
            SvrError::NoSform(p) => write!(f, "{} : pas de sform (affine absente)", p.display()),
            SvrError::DegenerateAffine(p) => {
                write!(f, "{} : affine dégénérée (colonne nulle ou non finie)", p.display())
            }
            SvrError::ShearedAffine { path, cosine } => write!(
                f,
                "{} : axes non orthogonaux (cosinus {cosine:.2e} > {COSINE_MAX:.0e})",
                path.display()
            ),
            SvrError::MaskGridMismatch { path, detail } => {
                write!(f, "{} : masque incompatible avec le stack ({detail})", path.display())
            }
            SvrError::EmptyMask(p) => write!(f, "{} : masque vide", p.display()),
            SvrError::NoMask(p) => write!(f, "{} : pas de masque cérébral attaché", p.display()),
            SvrError::InconsistentSpacing { path, pixdim, columns } => write!(
                f,
                "{} : pixdim {pixdim:?} incohérent avec les normes de l'affine {columns:?}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SvrError {}

/// Écart maximal toléré, en mm, entre l'affine d'un masque et celle de son stack.
const MASK_AFFINE_TOLERANCE_MM: f64 = 1e-3;

/// Masque cérébral d'un stack, **nettoyé** : seule la plus grande composante connexe est gardée.
///
/// Les petits îlots parasites (jusqu'à plusieurs % des voxels sur certains stacks) agrandissent les
/// boîtes englobantes et déplacent les barycentres ; la plus grande composante 3D (26 voisins) les écarte.
#[derive(Debug)]
pub struct BrainMask {
    voxels: Array3<bool>,
    discarded: usize,
    barycenter: Vector3<f64>,
}

impl BrainMask {
    /// Lit un masque NIfTI (tout voxel non nul est dans le masque), vérifie qu'il a la grille de
    /// `stack`, ne garde que la plus grande composante connexe et calcule son barycentre.
    ///
    /// # Erreurs
    /// [`SvrError::Read`] (illisible, non 3D) ; [`SvrError::MaskGridMismatch`] (dimensions ou affine
    /// différentes de celles du stack, ou `sform` absent) ; [`SvrError::EmptyMask`].
    pub fn read(path: &Path, stack: &Stack) -> Result<BrainMask, SvrError> {
        let lecture = |source| SvrError::Read { path: path.to_path_buf(), source };
        let info = volume_info(path).map_err(lecture)?;
        let incompatible = |detail: String| SvrError::MaskGridMismatch { path: path.to_path_buf(), detail };

        let dim: Vec<usize> = info.dim.iter().map(|&d| usize::from(d)).collect();
        let (nx, ny, nz) = stack.dim();
        if dim != [nx, ny, nz] {
            return Err(incompatible(format!("dimensions {dim:?} au lieu de {:?}", [nx, ny, nz])));
        }
        let lignes = info.affine.ok_or_else(|| incompatible("pas de sform".to_string()))?;
        let ecart = (0..4)
            .flat_map(|r| (0..4).map(move |c| (r, c)))
            .map(|(r, c)| (f64::from(lignes[r][c]) - stack.affine()[(r, c)]).abs())
            .fold(0.0_f64, f64::max);
        if ecart > MASK_AFFINE_TOLERANCE_MM {
            return Err(incompatible(format!("affine différente de celle du stack (écart {ecart:.2e})")));
        }

        let brut = read_volume(path).map_err(lecture)?.mapv(|v| v != 0.0);
        let total = brut.iter().filter(|&&v| v).count();
        if total == 0 {
            return Err(SvrError::EmptyMask(path.to_path_buf()));
        }
        let (voxels, gardes) = largest_component(&brut);
        let somme = voxels
            .indexed_iter()
            .filter(|(_, &v)| v)
            .fold(Vector3::zeros(), |s, ((i, j, k), _)| s + Vector3::new(i as f64, j as f64, k as f64));
        Ok(BrainMask { voxels, discarded: total - gardes, barycenter: somme / gardes as f64 })
    }

    /// Voxels du masque nettoyé, axes `[x, y, z]`, même grille que le stack.
    pub fn voxels(&self) -> &Array3<bool> {
        &self.voxels
    }

    /// Nombre de voxels du masque nettoyé.
    pub fn voxel_count(&self) -> usize {
        self.voxels.iter().filter(|&&v| v).count()
    }

    /// Nombre de voxels écartés (hors de la plus grande composante).
    pub fn discarded_count(&self) -> usize {
        self.discarded
    }

    /// Barycentre 3D du masque nettoyé, en **indices de voxel** continus `(i, j, k)`.
    pub fn barycenter_index(&self) -> Vector3<f64> {
        self.barycenter
    }
}

/// Plus grande composante connexe d'un masque 3D (26 voisins : faces, arêtes et coins). Rend le
/// masque de cette composante et son nombre de voxels. Parcours en largeur ; à taille égale, la
/// première rencontrée dans l'ordre `x`, `y`, `z` l'emporte (comme `scipy.ndimage.label`).
fn largest_component(masque: &Array3<bool>) -> (Array3<bool>, usize) {
    let (nx, ny, nz) = masque.dim();
    let dims = [nx as isize, ny as isize, nz as isize];
    let mut vu = Array3::<bool>::from_elem((nx, ny, nz), false);
    let mut meilleure: Vec<[usize; 3]> = Vec::new();
    for ((i, j, k), &dedans) in masque.indexed_iter() {
        if !dedans || vu[[i, j, k]] {
            continue;
        }
        // Le `Vec` sert de file : on ajoute à la fin, `lu` avance dans ce qui reste à visiter.
        let mut composante = vec![[i, j, k]];
        vu[[i, j, k]] = true;
        let mut lu = 0;
        while lu < composante.len() {
            let [a, b, c] = composante[lu];
            lu += 1;
            for da in -1_isize..=1 {
                for db in -1_isize..=1 {
                    for dc in -1_isize..=1 {
                        if (da, db, dc) == (0, 0, 0) {
                            continue;
                        }
                        // `isize` : a - 1 peut valoir -1, ce qu'un `usize` ne saurait représenter.
                        let v = [a as isize + da, b as isize + db, c as isize + dc];
                        if (0..3).any(|n| v[n] < 0 || v[n] >= dims[n]) {
                            continue;
                        }
                        let p = [v[0] as usize, v[1] as usize, v[2] as usize];
                        if masque[p] && !vu[p] {
                            vu[p] = true;
                            composante.push(p);
                        }
                    }
                }
            }
        }
        if composante.len() > meilleure.len() {
            meilleure = composante;
        }
    }
    let mut sortie = Array3::<bool>::from_elem((nx, ny, nz), false);
    for p in &meilleure {
        sortie[*p] = true;
    }
    (sortie, meilleure.len())
}

/// Écart maximal, en mm, entre les barycentres de masque de deux stacks voisins d'un même groupe.
///
/// **Provisoire** : choisi après avoir vu les données (dans le jeu de développement, plus grand écart au
/// sein d'un groupe : 13 mm ; plus petit écart entre groupes : 23 mm, soit 10 mm de marge seulement).
pub const DEFAULT_GROUP_GAP_MM: f64 = 18.0;

/// Résultat de [`group_stacks`].
#[derive(Debug, Clone, PartialEq)]
pub struct StackGroups {
    /// Groupes d'indices de stacks (indices dans la liste donnée), du plus grand au plus petit (à taille
    /// égale, celui dont le plus petit indice est le plus petit d'abord) ; indices croissants dans un groupe.
    pub groups: Vec<Vec<usize>>,
    /// Distance (mm) entre les barycentres 3D des masques de chaque paire de stacks ; matrice symétrique.
    pub distances: Vec<Vec<f64>>,
}

/// Regroupe des stacks qui partagent un repère, d'après les barycentres 3D de leurs masques cérébraux.
///
/// Deux stacks dont les barycentres sont à `max_gap_mm` ou moins sont dans le même groupe, **de proche en
/// proche** (chaînage : A proche de B et B proche de C met A, B et C ensemble, même si A et C sont plus
/// éloignés). Seules la géométrie et les masques interviennent : aucune information du JSON d'acquisition.
/// Un groupe d'un seul stack est possible.
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque attaché.
pub fn group_stacks(stacks: &[Stack], max_gap_mm: f64) -> Result<StackGroups, SvrError> {
    let barycentres: Vec<Vector3<f64>> = stacks
        .iter()
        .map(|s| s.brain_barycenter_world().ok_or_else(|| SvrError::NoMask(s.path().to_path_buf())))
        .collect::<Result<_, _>>()?;
    let n = barycentres.len();
    let distances: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| (barycentres[i] - barycentres[j]).norm()).collect())
        .collect();

    // Composantes connexes du graphe « distance <= max_gap_mm », par parcours en largeur (comme en 1d).
    let mut vu = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for depart in 0..n {
        if vu[depart] {
            continue;
        }
        vu[depart] = true;
        let mut groupe = vec![depart];
        let mut lu = 0;
        while lu < groupe.len() {
            let x = groupe[lu];
            lu += 1;
            for y in 0..n {
                if !vu[y] && distances[x][y] <= max_gap_mm {
                    vu[y] = true;
                    groupe.push(y);
                }
            }
        }
        groupe.sort_unstable();
        groups.push(groupe);
    }
    // Tri stable : plus grand groupe d'abord, puis plus petit indice.
    groups.sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]));
    Ok(StackGroups { groups, distances })
}

/// Boîte alignée sur les axes du monde (RAS+, mm) : le plus petit pavé qui contient un ensemble de points.
///
/// Pour un stack, c'est la boîte des 8 coins, en **centres de voxel** (pas de demi-voxel de marge).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Coin de coordonnées minimales.
    pub min: Vector3<f64>,
    /// Coin de coordonnées maximales.
    pub max: Vector3<f64>,
}

impl BoundingBox {
    /// Plus petite boîte contenant `self` et `autre`.
    pub fn union(&self, autre: &BoundingBox) -> BoundingBox {
        // `inf` / `sup` : minimum / maximum composante par composante.
        BoundingBox { min: self.min.inf(&autre.min), max: self.max.sup(&autre.max) }
    }

    /// Dimensions de la boîte (mm), selon x, y, z.
    pub fn size(&self) -> Vector3<f64> {
        self.max - self.min
    }
}

/// Lit plusieurs stacks, dans l'ordre donné. S'arrête à la première erreur, qui nomme le fichier fautif.
///
/// # Erreurs
/// La première erreur de [`Stack::read`].
pub fn read_stacks(chemins: &[PathBuf]) -> Result<Vec<Stack>, SvrError> {
    // `collect` sait rassembler des `Result<Stack, _>` en un `Result<Vec<Stack>, _>`.
    chemins.iter().map(|c| Stack::read(c)).collect()
}

/// Un stack : un volume 3D de coupes 2D, avec sa géométrie voxel → monde.
///
/// Les champs sont privés : un `Stack` ne se construit que par [`Stack::read`], qui vérifie la
/// géométrie. Tout `Stack` existant a donc une affine finie, de colonnes orthogonales et de
/// normes cohérentes avec `pixdim`.
#[derive(Debug)]
pub struct Stack {
    path: PathBuf,
    data: Array3<f32>,
    affine: Matrix4<f64>,
    spacing: [f64; 3],
    mask: Option<BrainMask>,
}

impl Stack {
    /// Lit un stack NIfTI (`.nii` ou `.nii.gz`) et valide sa géométrie.
    ///
    /// Les voxels sont lus en `f32`, axes `[x, y, z]`, facteur d'échelle de l'en-tête appliqué.
    /// L'affine est celle du `sform`, convertie en `f64` sans perte. L'espacement est la norme
    /// de chaque colonne de l'affine (la géométrie réelle) ; il doit coïncider avec `pixdim`
    /// à 1e-3 mm près. Un déterminant négatif est accepté.
    ///
    /// # Erreurs
    /// [`SvrError::Read`] (fichier illisible, non 3D) ; [`SvrError::NoSform`] ;
    /// [`SvrError::DegenerateAffine`] ; [`SvrError::ShearedAffine`] ;
    /// [`SvrError::InconsistentSpacing`].
    pub fn read(path: &Path) -> Result<Stack, SvrError> {
        // `map_err` rattache le chemin à l'erreur du core : sur un ensemble de stacks, on sait lequel échoue.
        let lecture = |source| SvrError::Read { path: path.to_path_buf(), source };
        let info = volume_info(path).map_err(lecture)?;
        let lignes = info.affine.ok_or_else(|| SvrError::NoSform(path.to_path_buf()))?;
        // `f64::from` : conversion exacte du `f32` de l'en-tête.
        let affine = Matrix4::from_fn(|r, c| f64::from(lignes[r][c]));

        // Partie linéaire 3x3 : colonne `a` = déplacement monde (mm) d'un pas d'indice sur l'axe `a`.
        let lineaire: Matrix3<f64> = affine.fixed_view::<3, 3>(0, 0).into_owned();
        let colonnes = [lineaire.column(0), lineaire.column(1), lineaire.column(2)];
        let normes = [colonnes[0].norm(), colonnes[1].norm(), colonnes[2].norm()];
        // `!(n > 0.0)` est vrai aussi pour NaN, ce que `n <= 0.0` ne détecterait pas.
        if normes.iter().any(|n| !(*n > 0.0) || !n.is_finite()) {
            return Err(SvrError::DegenerateAffine(path.to_path_buf()));
        }

        let mut cosinus_max = 0.0_f64;
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let cosinus = colonnes[a].dot(&colonnes[b]).abs() / (normes[a] * normes[b]);
            cosinus_max = cosinus_max.max(cosinus);
        }
        if cosinus_max > COSINE_MAX {
            return Err(SvrError::ShearedAffine { path: path.to_path_buf(), cosine: cosinus_max });
        }

        let pixdim = [
            f64::from(info.spacing[0]),
            f64::from(info.spacing[1]),
            f64::from(info.spacing[2]),
        ];
        if (0..3).any(|a| (pixdim[a] - normes[a]).abs() > SPACING_TOLERANCE_MM) {
            return Err(SvrError::InconsistentSpacing {
                path: path.to_path_buf(),
                pixdim,
                columns: normes,
            });
        }

        let data = read_volume(path).map_err(lecture)?;
        Ok(Stack { path: path.to_path_buf(), data, affine, spacing: normes, mask: None })
    }

    /// Chemin du fichier lu.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Voxels, axes `[x, y, z]` ; la coupe `k` est le plan `z = k`.
    pub fn data(&self) -> &Array3<f32> {
        &self.data
    }

    /// Affine voxel → monde (RAS+, mm), 4×4, coordonnées homogènes.
    pub fn affine(&self) -> &Matrix4<f64> {
        &self.affine
    }

    /// Taille d'un pas d'indice sur chaque axe, en mm (normes des colonnes de l'affine).
    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    /// Nombre de voxels sur chaque axe `(x, y, z)` ; `z` est le nombre de coupes.
    pub fn dim(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    /// Boîte englobante dans le monde (RAS+, mm) : minimum et maximum, axe par axe, des 8 coins du
    /// stack (centres des voxels `(0|nx-1, 0|ny-1, 0|nz-1)`).
    pub fn world_bounding_box(&self) -> BoundingBox {
        let (nx, ny, nz) = self.dim();
        let (mut min, mut max) = (Vector3::repeat(f64::INFINITY), Vector3::repeat(f64::NEG_INFINITY));
        for i in [0, nx - 1] {
            for j in [0, ny - 1] {
                for k in [0, nz - 1] {
                    let p = (self.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
                    min = min.inf(&p);
                    max = max.sup(&p);
                }
            }
        }
        BoundingBox { min, max }
    }

    /// Lit un masque cérébral, le nettoie (plus grande composante connexe) et l'attache à ce stack.
    /// En cas d'erreur, le stack reste inchangé (il n'est pas consommé : `&mut self`).
    ///
    /// # Erreurs
    /// Celles de [`BrainMask::read`].
    pub fn set_brain_mask(&mut self, chemin: &Path) -> Result<(), SvrError> {
        self.mask = Some(BrainMask::read(chemin, self)?);
        Ok(())
    }

    /// Barycentre 3D du masque cérébral, en monde (RAS+, mm), s'il y a un masque attaché.
    pub fn brain_barycenter_world(&self) -> Option<Vector3<f64>> {
        let b = self.mask.as_ref()?.barycenter_index();
        Some((self.affine * b.push(1.0)).xyz())
    }

    /// Masque cérébral attaché, s'il y en a un.
    pub fn brain_mask(&self) -> Option<&BrainMask> {
        self.mask.as_ref()
    }

    /// La coupe `k` (plan `z = k`), ou `None` si `k` dépasse le nombre de coupes.
    ///
    /// La coupe **emprunte** ce stack : le `'_` de `Slice<'_>` dit qu'elle ne peut pas lui survivre.
    pub fn slice(&self, k: usize) -> Option<Slice<'_>> {
        (k < self.data.dim().2).then_some(Slice { stack: self, k })
    }

    /// Toutes les coupes, dans l'ordre des `k` croissants.
    pub fn slices(&self) -> impl Iterator<Item = Slice<'_>> {
        (0..self.data.dim().2).map(move |k| Slice { stack: self, k })
    }
}

/// Une coupe 2D d'un [`Stack`] et son repère dans le monde.
///
/// `Slice<'a>` **emprunte** le stack (`&'a Stack`) : le compilateur garantit qu'une coupe ne vit
/// jamais plus longtemps que le stack d'où elle vient. Elle ne contient qu'une référence et un
/// indice, donc `Copy`.
///
/// Géométrie de la coupe `k` : un point de coordonnées `(i, j)` dans le plan de la coupe a pour
/// position monde `A · (i, j, k, 1)`, soit la matrice [`Slice::affine`] appliquée à `(i, j, 0, 1)`.
/// Aucune transformation de mouvement n'est incluse : elle viendra composée **après** cette matrice.
#[derive(Debug, Clone, Copy)]
pub struct Slice<'a> {
    stack: &'a Stack,
    k: usize,
}

impl<'a> Slice<'a> {
    /// Indice `k` de la coupe dans son stack.
    pub fn index(&self) -> usize {
        self.k
    }

    /// Nombre de voxels de la coupe `(x, y)`.
    pub fn dim(&self) -> (usize, usize) {
        let (nx, ny, _) = self.stack.data.dim();
        (nx, ny)
    }

    /// Voxels de la coupe, vue en lecture seule sur les données du stack.
    ///
    /// La vue vit aussi longtemps que le **stack** (`'a`), pas seulement que cette `Slice`.
    pub fn data(&self) -> ArrayView2<'a, f32> {
        self.stack.data.index_axis(Axis(2), self.k)
    }

    /// Affine de la coupe : `A · T(0, 0, k)`. Envoie `(i, j, 0, 1)` sur la position monde (mm) du
    /// voxel `(i, j)` de cette coupe. La translation de `k` pas le long de la 3ᵉ colonne s'ajoute à
    /// l'origine : `A · T(0,0,k)` a la même partie linéaire que `A`, avec `origine + k · colonne 3`.
    pub fn affine(&self) -> Matrix4<f64> {
        let mut m = *self.stack.affine();
        let k = self.k as f64;
        for r in 0..3 {
            m[(r, 3)] += k * m[(r, 2)];
        }
        m
    }

    /// Position monde (RAS+, mm) du point `(i, j)` du plan de la coupe (indices continus permis).
    pub fn pixel_to_world(&self, i: f64, j: f64) -> Vector3<f64> {
        (self.affine() * Vector4::new(i, j, 0.0, 1.0)).xyz()
    }

    /// Axes du plan de la coupe : colonnes 0 et 1 de l'affine, normalisées (vecteurs unitaires).
    pub fn in_plane_axes(&self) -> [Vector3<f64>; 2] {
        let a = self.stack.affine();
        [
            a.fixed_view::<3, 1>(0, 0).into_owned().normalize(),
            a.fixed_view::<3, 1>(0, 1).into_owned().normalize(),
        ]
    }

    /// Normale de la coupe : **3ᵉ colonne de l'affine, normalisée**, orientée dans le sens des `k`
    /// croissants. Ce n'est volontairement pas le produit vectoriel des axes du plan, qui pointerait
    /// à l'envers pour une affine à déterminant négatif (le cas des stacks du jeu de développement).
    pub fn normal(&self) -> Vector3<f64> {
        self.stack.affine().fixed_view::<3, 1>(0, 2).into_owned().normalize()
    }

    /// Épaisseur de la coupe en mm : l'espacement entre coupes (NIfTI-1 n'a pas de champ
    /// d'épaisseur ; hypothèse vérifiée sans écart sur les 96 stacks du jeu de développement).
    pub fn thickness(&self) -> f64 {
        self.stack.spacing()[2]
    }

    /// **Pivot P3** de la coupe : le barycentre 3D du masque cérébral, pris dans le plan de la coupe
    /// (les coordonnées `(i, j)` du barycentre, avec `k` pour la 3ᵉ), en monde (mm). `None` si aucun
    /// masque n'est attaché au stack : le repli éventuel sur [`Slice::geometric_center`] est une
    /// décision de l'appelant, jamais silencieuse.
    ///
    /// Suppose un mouvement modéré du fœtus *dans* le stack : le barycentre 3D est le même pour toutes les coupes.
    pub fn brain_pivot(&self) -> Option<Vector3<f64>> {
        let b = self.stack.mask.as_ref()?.barycenter_index();
        Some(self.pixel_to_world(b.x, b.y))
    }

    /// Centre géométrique de la coupe (monde, mm) : le point `((nx-1)/2, (ny-1)/2)`. Référence
    /// « sans masque » ; le pivot choisi (barycentre du masque) est calculé à l'étape suivante.
    pub fn geometric_center(&self) -> Vector3<f64> {
        let (nx, ny) = self.dim();
        self.pixel_to_world((nx - 1) as f64 / 2.0, (ny - 1) as f64 / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};
    use nifti::{writer::WriterOptions, NiftiHeader};

    fn racine() -> String {
        format!("{}/../..", env!("CARGO_MANIFEST_DIR"))
    }

    /// Dossier temporaire propre à un test (nom + processus), supprimé en fin de test.
    struct Temp(PathBuf);
    impl Temp {
        fn new(nom: &str) -> Temp {
            let d = std::env::temp_dir().join(format!("medoxide_svr_{nom}_{}", std::process::id()));
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

    /// En-tête synthétique : `sform` donné par ses 3 premières lignes, `pixdim` donné.
    fn en_tete(lignes: [[f32; 4]; 3], pixdim: [f32; 3], sform_code: i16) -> NiftiHeader {
        let mut h = NiftiHeader::default();
        h.sform_code = sform_code;
        h.srow_x = lignes[0];
        h.srow_y = lignes[1];
        h.srow_z = lignes[2];
        h.pixdim = [1.0, pixdim[0], pixdim[1], pixdim[2], 1.0, 1.0, 1.0, 1.0];
        h
    }

    fn ecrire_3d(chemin: &Path, h: &NiftiHeader) {
        let data = Array3::<f32>::from_elem((4, 3, 2), 1.0);
        WriterOptions::new(chemin).reference_header(h).write_nifti(&data).unwrap();
    }

    const DIAG: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, 20.0], [0.0, 0.0, 3.0, 30.0]];

    #[test]
    fn reads_a_valid_synthetic_stack() {
        let t = Temp::new("valide");
        let f = t.fichier("s.nii.gz");
        ecrire_3d(&f, &en_tete(DIAG, [1.0, 2.0, 3.0], 1));
        let s = Stack::read(&f).unwrap();
        assert_eq!(s.dim(), (4, 3, 2));
        assert_eq!(s.spacing(), [1.0, 2.0, 3.0]);
        assert_eq!(s.affine()[(0, 3)], 10.0);
        assert_eq!(s.affine()[(3, 3)], 1.0);
    }

    /// Un déterminant négatif n'est pas une erreur (cas de tous les stacks du jeu de développement).
    #[test]
    fn accepts_a_left_handed_affine() {
        let t = Temp::new("gauche");
        let f = t.fichier("s.nii.gz");
        let lignes = [[-1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]];
        ecrire_3d(&f, &en_tete(lignes, [1.0, 2.0, 3.0], 1));
        let s = Stack::read(&f).unwrap();
        assert!(s.affine().fixed_view::<3, 3>(0, 0).determinant() < 0.0);
        assert_eq!(s.spacing(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_a_missing_sform() {
        let t = Temp::new("sans_sform");
        let f = t.fichier("s.nii.gz");
        ecrire_3d(&f, &en_tete(DIAG, [1.0, 2.0, 3.0], 0));
        assert!(matches!(Stack::read(&f), Err(SvrError::NoSform(_))));
    }

    #[test]
    fn rejects_a_sheared_affine() {
        let t = Temp::new("cisaille");
        let f = t.fichier("s.nii.gz");
        // 2e colonne = (0,1,0) + 0,2 × 1re colonne : cosinus entre colonnes ≈ 0,2.
        let lignes = [[1.0, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let norme = (1.0_f32 + 0.04).sqrt();
        ecrire_3d(&f, &en_tete(lignes, [1.0, norme, 1.0], 1));
        match Stack::read(&f) {
            Err(SvrError::ShearedAffine { cosine, .. }) => assert!(cosine > 0.1, "{cosine}"),
            autre => panic!("attendu ShearedAffine, obtenu {autre:?}"),
        }
    }

    #[test]
    fn rejects_inconsistent_pixdim() {
        let t = Temp::new("pixdim");
        let f = t.fichier("s.nii.gz");
        ecrire_3d(&f, &en_tete(DIAG, [1.0, 2.0, 3.5], 1)); // l'affine dit 3, pixdim dit 3,5
        assert!(matches!(Stack::read(&f), Err(SvrError::InconsistentSpacing { .. })));
    }

    #[test]
    fn rejects_a_degenerate_affine() {
        let t = Temp::new("degenere");
        let f = t.fichier("s.nii.gz");
        let lignes = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]; // colonne 1 nulle
        ecrire_3d(&f, &en_tete(lignes, [1.0, 0.0, 1.0], 1));
        assert!(matches!(Stack::read(&f), Err(SvrError::DegenerateAffine(_))));
    }

    #[test]
    fn rejects_a_4d_file_and_a_missing_file() {
        let t = Temp::new("erreurs");
        let f = t.fichier("s4d.nii.gz");
        let data = Array4::<f32>::from_elem((4, 3, 2, 2), 1.0);
        WriterOptions::new(&f)
            .reference_header(&en_tete(DIAG, [1.0, 2.0, 3.0], 1))
            .write_nifti(&data)
            .unwrap();
        assert!(matches!(
            Stack::read(&f),
            Err(SvrError::Read { source: CoreError::NotVolume3D(_), .. })
        ));
        assert!(matches!(
            Stack::read(&t.fichier("absent.nii.gz")),
            Err(SvrError::Read { source: CoreError::Nifti(_), .. })
        ));
    }

    /// Affines de référence de nibabel (`img.affine`) pour 3 volumes de Fetal-BET, arrondis à 4
    /// décimales : mêmes valeurs que dans `medoxide-core`, lues cette fois par `Stack`.
    #[test]
    fn reads_fetal_bet_volumes_with_nibabel_affines() {
        let attendu: [(&str, [[f64; 4]; 3]); 3] = [
            ("fetus_03", [[-1.1719, 0.0, 0.0, 155.1199], [0.0, -1.0771, 0.7879, 58.3842], [0.0, 0.4617, 1.8383, -254.6115]]),
            ("fetus_06", [[-0.8439, -0.147, -0.8127, 141.5291], [0.4083, -0.3038, -1.6797, -19.843], [0.0, -0.8747, 0.7199, -69.1684]]),
            ("fetus_12", [[-0.5306, -0.4462, 1.8271, 96.2711], [-0.866, 0.2733, -1.1193, 46.4229], [0.0, -0.8705, -1.288, -12.331]]),
        ];
        for (nom, voulu) in attendu {
            let s = Stack::read(Path::new(&format!("{}/data/sourcedata/{nom}.nii.gz", racine()))).unwrap();
            for r in 0..3 {
                for c in 0..4 {
                    assert!((s.affine()[(r, c)] - voulu[r][c]).abs() < 1e-3, "{nom} ({r},{c})");
                }
            }
            assert_eq!(s.affine().row(3).iter().copied().collect::<Vec<_>>(), vec![0.0, 0.0, 0.0, 1.0]);
        }
    }

    /// Parcourt récursivement un dossier et collecte les fichiers dont le nom se termine par `fin`.
    fn trouver(dossier: &Path, fin: &str, sortie: &mut Vec<PathBuf>) {
        for entree in std::fs::read_dir(dossier).unwrap().flatten() {
            let chemin = entree.path();
            if chemin.is_dir() {
                trouver(&chemin, fin, sortie);
            } else if chemin.to_string_lossy().ends_with(fin) {
                sortie.push(chemin);
            }
        }
    }

    /// Jeu de développement : les 96 stacks se chargent, tous à déterminant négatif, géométrie
    /// valide (orthogonale, pixdim cohérent). Données locales (`data/`, hors de Git).
    #[test]
    fn reads_all_development_stacks_even_left_handed() {
        let mut fichiers = Vec::new();
        trouver(&Path::new(&racine()).join("data/svr/jeu_reel_tru_haste"), "_T2w.nii.gz", &mut fichiers);
        fichiers.retain(|f| !f.to_string_lossy().contains("/derivatives/") && !f.to_string_lossy().contains("/sourcedata/"));
        assert_eq!(fichiers.len(), 96, "jeu de développement incomplet");
        let mut gauches = 0;
        for f in &fichiers {
            let s = Stack::read(f).unwrap_or_else(|e| panic!("{e}"));
            gauches += usize::from(s.affine().fixed_view::<3, 3>(0, 0).determinant() < 0.0);
            assert_eq!(s.dim().2, s.data().dim().2);
        }
        assert_eq!(gauches, 96, "tous les stacks du jeu sont censés être à déterminant négatif");
    }

    // ------------------------------------------------------------------ étape 1b : coupes

    /// Stack synthétique 4×3×2 d'affine donnée (3 premières lignes).
    fn stack_synthetique(t: &Temp, lignes: [[f32; 4]; 3], pixdim: [f32; 3]) -> Stack {
        let f = t.fichier("coupes.nii.gz");
        ecrire_3d(&f, &en_tete(lignes, pixdim, 1));
        Stack::read(&f).unwrap()
    }

    #[test]
    fn slice_geometry_of_a_diagonal_stack() {
        let t = Temp::new("coupe_diag");
        let s = stack_synthetique(&t, DIAG, [1.0, 2.0, 3.0]);
        let c = s.slice(1).unwrap();
        // origine (10, 20, 30) ; k = 1 décale de 3 mm selon z : (10, 20, 33).
        assert_eq!(c.pixel_to_world(0.0, 0.0), Vector3::new(10.0, 20.0, 33.0));
        assert_eq!(c.pixel_to_world(3.0, 2.0), Vector3::new(13.0, 24.0, 33.0));
        // centre géométrique : ((4-1)/2, (3-1)/2) = (1,5 ; 1) -> (11,5 ; 22 ; 33).
        assert_eq!(c.geometric_center(), Vector3::new(11.5, 22.0, 33.0));
        assert_eq!(c.normal(), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(c.thickness(), 3.0);
        assert_eq!(c.in_plane_axes()[0], Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(c.data().dim(), (4, 3));
        assert!(s.slice(2).is_none());
        assert_eq!(s.slices().count(), 2);
        assert_eq!(s.slices().map(|c| c.index()).collect::<Vec<_>>(), vec![0, 1]);
    }

    /// La normale est la 3ᵉ colonne, pas le produit vectoriel des axes du plan (opposé ici).
    #[test]
    fn normal_follows_the_third_column_for_a_left_handed_affine() {
        let t = Temp::new("coupe_gauche");
        let lignes = [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let s = stack_synthetique(&t, lignes, [1.0, 1.0, 1.0]);
        let c = s.slice(0).unwrap();
        let [u, v] = c.in_plane_axes();
        assert_eq!(c.normal(), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(u.cross(&v), Vector3::new(0.0, 0.0, -1.0)); // le produit vectoriel pointerait à l'envers
        assert!(u.cross(&v).dot(&c.normal()) < 0.0);
    }

    /// Lit la table de références produite par `scripts/make_reference_slice_coords.py` :
    /// (chemin du stack, lignes `[i, j, k, x_nibabel, y, z, x_simpleitk_ras, y, z]`).
    fn references() -> Vec<(PathBuf, Vec<[f64; 9]>)> {
        let racine = racine();
        let index = std::fs::read_to_string(format!("{racine}/data/reference/slicecoords/index.tsv"))
            .expect("références absentes : lancer scripts/make_reference_slice_coords.py");
        index
            .lines()
            .map(|l| {
                let (stack, reference) = l.split_once('\t').unwrap();
                let octets = std::fs::read(format!("{racine}/{reference}")).unwrap();
                let valeurs: Vec<f64> =
                    octets.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
                let lignes = valeurs.chunks_exact(9).map(|r| <[f64; 9]>::try_from(r).unwrap()).collect();
                (PathBuf::from(format!("{racine}/{stack}")), lignes)
            })
            .collect()
    }

    /// Critères 1 et 2 de l'étape 1 : coordonnées monde par coupe contre nibabel (< 1e-6 mm,
    /// calcul indépendant) et contre SimpleITK converti en RAS (< 1e-3 mm), sur les 8 volumes de
    /// Fetal-BET et les 96 stacks du jeu de développement.
    ///
    /// Seuil SimpleITK : 1e-3 mm au lieu des 1e-4 mm fixés d'abord. Écart mesuré 2,6e-4 mm : ITK
    /// bâtit sa géométrie avec `pixdim` (f32) et une direction orthonormalisée, d'où un écart de
    /// l'ordre du trois-millième de voxel avec l'affine stockée (qu'applique nibabel, et nous).
    #[test]
    fn world_coordinates_match_nibabel_and_simpleitk() {
        let refs = references();
        assert_eq!(refs.len(), 104, "8 volumes de Fetal-BET + 96 stacks");
        let (mut pire_nib, mut pire_itk) = (0.0_f64, 0.0_f64);
        for (chemin, lignes) in &refs {
            let stack = Stack::read(chemin).unwrap();
            let (_, _, nz) = stack.dim();
            for (n, r) in lignes.iter().enumerate() {
                let (i, j, k) = (r[0], r[1], r[2]);
                // 5 premiers points de chaque coupe : coins et centre, via la coupe ; le reste : points
                // continus dans le volume, via l'affine du stack.
                let monde = if n < 5 * nz {
                    stack.slice(k as usize).unwrap().pixel_to_world(i, j)
                } else {
                    (stack.affine() * Vector4::new(i, j, k, 1.0)).xyz()
                };
                let nib = Vector3::new(r[3], r[4], r[5]);
                let itk = Vector3::new(r[6], r[7], r[8]);
                pire_nib = pire_nib.max((monde - nib).amax());
                pire_itk = pire_itk.max((monde - itk).amax());
            }
        }
        println!("écart max : nibabel {pire_nib:.2e} mm, SimpleITK-RAS {pire_itk:.2e} mm");
        assert!(pire_nib < 1e-6, "nibabel : {pire_nib:.2e} mm");
        assert!(pire_itk < 1e-3, "SimpleITK : {pire_itk:.2e} mm");
    }

    /// Critère 3 : invariants géométriques de chaque stack, sans référence externe.
    #[test]
    fn slice_invariants_hold_on_all_stacks() {
        let refs = references();
        let mut n_coupes = 0;
        for (chemin, _) in &refs {
            let stack = Stack::read(chemin).unwrap();
            let pixdim = volume_info(chemin).unwrap().spacing.map(f64::from);
            let col3 = stack.affine().fixed_view::<3, 1>(0, 2).into_owned();
            let coupes: Vec<_> = stack.slices().collect();
            for c in &coupes {
                n_coupes += 1;
                let [u, v] = c.in_plane_axes();
                let n = c.normal();
                assert!((n.norm() - 1.0).abs() < 1e-12 && (u.norm() - 1.0).abs() < 1e-12);
                assert!(n.cross(&col3).norm() < 1e-9, "normale non colinéaire à la 3e colonne");
                assert!(n.dot(&col3) > 0.0, "normale à l'envers");
                assert!(n.dot(&u).abs() < 1e-6 && n.dot(&v).abs() < 1e-6, "normale non orthogonale au plan");
                // pas dans le plan = espacement (norme des colonnes) ≈ pixdim de l'en-tête (1e-4 mm)
                let pas_i = (c.pixel_to_world(1.0, 0.0) - c.pixel_to_world(0.0, 0.0)).norm();
                let pas_j = (c.pixel_to_world(0.0, 1.0) - c.pixel_to_world(0.0, 0.0)).norm();
                assert!((pas_i - pixdim[0]).abs() < 1e-4 && (pas_j - pixdim[1]).abs() < 1e-4);
                assert!((c.thickness() - pixdim[2]).abs() < 1e-4);
            }
            // centres de coupes consécutives : écart = espacement entre coupes, le long de la normale
            for paire in coupes.windows(2) {
                let ecart = paire[1].geometric_center() - paire[0].geometric_center();
                let n = paire[0].normal();
                assert!((ecart.dot(&n) - pixdim[2]).abs() < 1e-4, "pas entre coupes");
                assert!((ecart - n * ecart.dot(&n)).norm() < 1e-4, "décalage latéral entre coupes");
            }
        }
        println!("invariants vérifiés sur {n_coupes} coupes de {} stacks", refs.len());
        assert!(n_coupes > 3000);
    }

    // ------------------------------------------------------------------ étape 1c : ensembles de stacks

    #[test]
    fn bounding_box_of_synthetic_stacks() {
        let t = Temp::new("boite");
        // DIAG : pas (1, 2, 3), origine (10, 20, 30), 4×3×2 voxels -> x 10..13, y 20..24, z 30..33.
        let s = stack_synthetique(&t, DIAG, [1.0, 2.0, 3.0]);
        let b = s.world_bounding_box();
        assert_eq!(b.min, Vector3::new(10.0, 20.0, 30.0));
        assert_eq!(b.max, Vector3::new(13.0, 24.0, 33.0));
        assert_eq!(b.size(), Vector3::new(3.0, 4.0, 3.0));
        // Axe x inversé (déterminant négatif) : la boîte reste ordonnée, x de -3 à 0.
        let t2 = Temp::new("boite_gauche");
        let lignes = [[-1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 20.0], [0.0, 0.0, 3.0, 30.0]];
        let g = stack_synthetique(&t2, lignes, [1.0, 2.0, 3.0]).world_bounding_box();
        assert_eq!((g.min.x, g.max.x), (-3.0, 0.0));
        assert!(g.min.x < g.max.x && g.min.y < g.max.y && g.min.z < g.max.z);
        let u = b.union(&g);
        assert_eq!(u.min, Vector3::new(-3.0, 20.0, 30.0));
        assert_eq!(u.max, Vector3::new(13.0, 24.0, 33.0));
        assert_eq!(u.union(&b), u);
    }

    #[test]
    fn read_stacks_keeps_order_and_names_the_failing_file() {
        let t = Temp::new("ensemble");
        let (a, b) = (t.fichier("a.nii.gz"), t.fichier("b.nii.gz"));
        ecrire_3d(&a, &en_tete(DIAG, [1.0, 2.0, 3.0], 1));
        ecrire_3d(&b, &en_tete(DIAG, [1.0, 2.0, 3.0], 1));
        let v = read_stacks(&[b.clone(), a.clone()]).unwrap();
        assert_eq!((v[0].path(), v[1].path()), (b.as_path(), a.as_path()));
        // un fichier manquant au milieu : l'erreur nomme ce fichier, quel que soit son rang
        let absent = t.fichier("absent.nii.gz");
        let e = read_stacks(&[a.clone(), absent.clone(), b]).unwrap_err();
        assert!(matches!(&e, SvrError::Read { path, .. } if *path == absent), "{e}");
        assert!(e.to_string().contains("absent.nii.gz"), "{e}");
        assert!(read_stacks(&[]).unwrap().is_empty());
    }

    /// Boîtes de l'étape 1c contre nibabel : les coins de la première et de la dernière coupe, déjà
    /// dans la table de références (calcul indépendant), donnent la boîte attendue (< 1e-6 mm).
    #[test]
    fn bounding_boxes_match_nibabel_corners() {
        let mut pire = 0.0_f64;
        for (chemin, lignes) in references() {
            let stack = Stack::read(&chemin).unwrap();
            let nz = stack.dim().2;
            // 5 points par coupe : les 4 premiers sont les coins (le 5e est le centre).
            let coins: Vec<Vector3<f64>> = lignes[..5 * nz]
                .iter()
                .enumerate()
                .filter(|(n, _)| n % 5 != 4)
                .map(|(_, r)| Vector3::new(r[3], r[4], r[5]))
                .collect();
            let min = coins.iter().fold(Vector3::repeat(f64::INFINITY), |m, p| m.inf(p));
            let max = coins.iter().fold(Vector3::repeat(f64::NEG_INFINITY), |m, p| m.sup(p));
            let b = stack.world_bounding_box();
            pire = pire.max((b.min - min).amax()).max((b.max - max).amax());
        }
        println!("écart max des boîtes englobantes : {pire:.2e} mm");
        assert!(pire < 1e-6, "{pire:.2e} mm");
    }

    /// Les 15 sujets du jeu de développement : leurs stacks se lisent d'un coup, et la boîte de
    /// l'ensemble contient celle de chaque stack. (Une boîte englobante est un indicateur faible de
    /// cohérence entre stacks : voir l'étude 02.)
    #[test]
    fn reads_each_development_subject_as_a_set() {
        let mut fichiers = Vec::new();
        trouver(&Path::new(&racine()).join("data/svr/jeu_reel_tru_haste"), "_T2w.nii.gz", &mut fichiers);
        fichiers.retain(|f| !f.to_string_lossy().contains("/derivatives/") && !f.to_string_lossy().contains("/sourcedata/"));
        let mut par_sujet: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
        for f in fichiers {
            let sujet = f.strip_prefix(Path::new(&racine()).join("data/svr/jeu_reel_tru_haste")).unwrap().components().next().unwrap().as_os_str().to_string_lossy().into_owned();
            par_sujet.entry(sujet).or_default().push(f);
        }
        assert_eq!(par_sujet.len(), 15);
        for (sujet, chemins) in &par_sujet {
            let stacks = read_stacks(chemins).unwrap_or_else(|e| panic!("{sujet} : {e}"));
            assert_eq!(stacks.len(), chemins.len());
            let tout = stacks.iter().map(|s| s.world_bounding_box()).reduce(|a, b| a.union(&b)).unwrap();
            for s in &stacks {
                let b = s.world_bounding_box();
                assert!(b.min >= tout.min && b.max <= tout.max, "{sujet}");
            }
        }
    }

    // ------------------------------------------------------------------ étape 1d : pivot P3

    /// Écrit un stack synthétique 6×5×3 (affine `DIAG`) et un masque `u8` de même grille, dont les voxels
    /// à 1 sont donnés ; rend les deux chemins.
    fn stack_et_masque(t: &Temp, voxels: &[[usize; 3]], masque_header: Option<NiftiHeader>) -> (PathBuf, PathBuf) {
        let h = en_tete(DIAG, [1.0, 2.0, 3.0], 1);
        let (f, m) = (t.fichier("stack.nii.gz"), t.fichier("masque.nii.gz"));
        WriterOptions::new(&f).reference_header(&h).write_nifti(&Array3::<f32>::from_elem((6, 5, 3), 1.0)).unwrap();
        let mut masque = Array3::<u8>::zeros((6, 5, 3));
        for v in voxels {
            masque[*v] = 1;
        }
        WriterOptions::new(&m).reference_header(&masque_header.unwrap_or(h)).write_nifti(&masque).unwrap();
        (f, m)
    }

    #[test]
    fn mask_keeps_the_largest_component_with_26_neighbours() {
        let t = Temp::new("masque_composantes");
        // bloc 2×2×2 en (1..3, 1..3, 0..2), un voxel qui ne le touche que par un COIN (3,3,2), un îlot (5,4,2)
        let mut voxels = Vec::new();
        for i in 1..3 { for j in 1..3 { for k in 0..2 { voxels.push([i, j, k]); } } }
        voxels.push([3, 3, 2]);
        voxels.push([5, 4, 2]);
        let (f, m) = stack_et_masque(&t, &voxels, None);
        let mut s = Stack::read(&f).unwrap();
        assert!(s.brain_mask().is_none());
        s.set_brain_mask(&m).unwrap();
        let masque = s.brain_mask().unwrap();
        assert_eq!(masque.voxel_count(), 9, "bloc de 8 + voxel en contact par un coin (26 voisins)");
        assert_eq!(masque.discarded_count(), 1, "l'îlot est écarté");
        assert!(!masque.voxels()[[5, 4, 2]] && masque.voxels()[[3, 3, 2]]);
        // barycentre : i = (8 × 1,5 + 3) / 9, j idem, k = (8 × 0,5 + 2) / 9
        let b = masque.barycenter_index();
        assert!((b - Vector3::new(15.0 / 9.0, 15.0 / 9.0, 6.0 / 9.0)).norm() < 1e-12, "{b:?}");
    }

    #[test]
    fn brain_pivot_lies_in_each_slice_plane_at_the_3d_barycentre() {
        let t = Temp::new("pivot");
        let (f, m) = stack_et_masque(&t, &[[1, 1, 0], [2, 1, 0], [1, 2, 1], [2, 2, 1]], None);
        let mut s = Stack::read(&f).unwrap();
        assert!(s.slice(0).unwrap().brain_pivot().is_none(), "sans masque : pas de pivot, pas de repli silencieux");
        s.set_brain_mask(&m).unwrap();
        // barycentre en indices : (1,5 ; 1,5 ; 0,5). Affine DIAG : pas (1, 2, 3), origine (10, 20, 30).
        let (p0, p1) = (s.slice(0).unwrap().brain_pivot().unwrap(), s.slice(1).unwrap().brain_pivot().unwrap());
        assert!((p0 - Vector3::new(11.5, 23.0, 30.0)).norm() < 1e-12, "{p0:?}");
        assert!((p1 - Vector3::new(11.5, 23.0, 33.0)).norm() < 1e-12, "{p1:?}");
        // mêmes (x, y) pour toutes les coupes, décalage d'une coupe = 3e colonne de l'affine
        assert!(((p1 - p0) - Vector3::new(0.0, 0.0, 3.0)).norm() < 1e-12);
        // le pivot est dans le plan de la coupe : écart au centre de coupe orthogonal à la normale... d'abord le plan
        let c = s.slice(1).unwrap();
        assert!((p1 - c.geometric_center()).dot(&c.normal()).abs() < 1e-12);
    }

    #[test]
    fn mask_errors_leave_the_stack_unchanged() {
        let t = Temp::new("masque_erreurs");
        // masque vide
        let (f, m) = stack_et_masque(&t, &[], None);
        let mut s = Stack::read(&f).unwrap();
        assert!(matches!(s.set_brain_mask(&m), Err(SvrError::EmptyMask(_))));
        assert!(s.brain_mask().is_none(), "échec : le stack n'a pas de masque");
        // affine décalée de 1 mm
        let mut h = en_tete(DIAG, [1.0, 2.0, 3.0], 1);
        h.srow_x[3] += 1.0;
        let (_, m2) = stack_et_masque(&t, &[[1, 1, 1]], Some(h));
        assert!(matches!(s.set_brain_mask(&m2), Err(SvrError::MaskGridMismatch { .. })));
        // dimensions différentes (4×3×2 au lieu de 6×5×3)
        let m3 = t.fichier("petit.nii.gz");
        ecrire_3d(&m3, &en_tete(DIAG, [1.0, 2.0, 3.0], 1));
        let e = s.set_brain_mask(&m3).unwrap_err();
        assert!(matches!(&e, SvrError::MaskGridMismatch { detail, .. } if detail.contains("dimensions")), "{e}");
        // sform absent
        let m4 = t.fichier("sans_sform.nii.gz");
        ecrire_3d(&m4, &en_tete(DIAG, [1.0, 2.0, 3.0], 0));
        assert!(s.set_brain_mask(&m4).is_err());
        // fichier absent
        assert!(matches!(s.set_brain_mask(&t.fichier("absent.nii.gz")), Err(SvrError::Read { .. })));
        assert!(s.brain_mask().is_none());
    }

    /// Lit un TSV de références : lignes de colonnes séparées par des tabulations.
    fn lire_tsv(chemin: &str) -> Vec<Vec<String>> {
        std::fs::read_to_string(format!("{}/{chemin}", racine()))
            .expect("références absentes : lancer scripts/make_reference_mask_pivots.py")
            .lines()
            .map(|l| l.split('\t').map(str::to_string).collect())
            .collect()
    }

    /// Vérifie, sur un stack de `pas` en `pas` parmi les 96 du jeu, le nettoyage du masque (même nombre de
    /// voxels que `scipy.ndimage.label`, 26 voisins), le barycentre (égal à celui de numpy, < 1e-9 voxel)
    /// et le pivot P3 de chaque coupe (égal à `A @ (bi, bj, k, 1)`, < 1e-6 mm).
    fn verifier_masques_et_pivots(pas: usize) {
        let racine = racine();
        let mut pivots: std::collections::HashMap<(String, usize), [f64; 3]> = Default::default();
        for c in lire_tsv("data/reference/pivots/pivots.tsv") {
            let xyz = [c[2].parse().unwrap(), c[3].parse().unwrap(), c[4].parse().unwrap()];
            pivots.insert((c[0].clone(), c[1].parse().unwrap()), xyz);
        }
        let (mut pire_b, mut pire_p, mut n_ecartes, mut n_pivots, mut n_stacks) = (0.0_f64, 0.0_f64, 0usize, 0usize, 0usize);
        let masques = lire_tsv("data/reference/pivots/masques.tsv");
        assert_eq!(masques.len(), 96);
        for c in masques.iter().step_by(pas) {
            n_stacks += 1;
            let mut s = Stack::read(Path::new(&format!("{racine}/{}", c[0]))).unwrap();
            s.set_brain_mask(Path::new(&format!("{racine}/{}", c[1]))).unwrap();
            let m = s.brain_mask().unwrap();
            let (total, gardes): (usize, usize) = (c[2].parse().unwrap(), c[3].parse().unwrap());
            assert_eq!(m.voxel_count(), gardes, "{} : voxels conservés", c[0]);
            assert_eq!(m.discarded_count(), total - gardes, "{} : voxels écartés", c[0]);
            n_ecartes += total - gardes;
            let attendu = Vector3::new(c[4].parse().unwrap(), c[5].parse().unwrap(), c[6].parse().unwrap());
            pire_b = pire_b.max((m.barycenter_index() - attendu).amax());
            for coupe in s.slices() {
                let [x, y, z] = pivots[&(c[0].clone(), coupe.index())];
                pire_p = pire_p.max((coupe.brain_pivot().unwrap() - Vector3::new(x, y, z)).amax());
                n_pivots += 1;
            }
        }
        println!("{n_stacks} stacks : barycentre écart max {pire_b:.2e} voxel ; {n_pivots} pivots écart max {pire_p:.2e} mm ; {n_ecartes} voxels écartés");
        assert!(pire_b < 1e-9 && pire_p < 1e-6);
    }

    /// Version rapide (un stack sur 8) de l'étape 1d.
    #[test]
    fn mask_cleaning_and_pivots_match_scipy_and_numpy_sampled() {
        verifier_masques_et_pivots(8);
    }

    /// Version complète : les 96 stacks (plus d'une minute en debug). Lancer avec :
    /// `cargo test -p medoxide-svr -- --ignored --nocapture mask_cleaning_and_pivots_match_scipy_and_numpy_all`
    #[test]
    #[ignore = "long : voir la doc du test"]
    fn mask_cleaning_and_pivots_match_scipy_and_numpy_all() {
        verifier_masques_et_pivots(1);
    }

    // ------------------------------------------------------------------ étape 1e : groupes de stacks

    /// Stack synthétique 6×5×3 d'origine x donnée, avec un masque d'un seul voxel : son barycentre monde est
    /// `origine_x + 1` selon x (indice 1, pas de 1 mm).
    fn stack_masque_en(t: &Temp, nom: &str, origine_x: f32) -> Stack {
        let mut h = en_tete(DIAG, [1.0, 2.0, 3.0], 1);
        h.srow_x[3] = origine_x;
        let (f, m) = (t.fichier(&format!("{nom}.nii.gz")), t.fichier(&format!("{nom}_masque.nii.gz")));
        WriterOptions::new(&f).reference_header(&h).write_nifti(&Array3::<f32>::from_elem((6, 5, 3), 1.0)).unwrap();
        let mut masque = Array3::<u8>::zeros((6, 5, 3));
        masque[[1, 1, 1]] = 1;
        WriterOptions::new(&m).reference_header(&h).write_nifti(&masque).unwrap();
        let mut s = Stack::read(&f).unwrap();
        s.set_brain_mask(&m).unwrap();
        s
    }

    #[test]
    fn groups_chain_neighbours_and_order_by_size() {
        let t = Temp::new("groupes");
        // x = 100, 0, 105, 10, 20 : {0, 10, 20} se chaînent (0-10, 10-20) bien que 0 et 20 soient à 20 mm ;
        // {100, 105} forment un autre groupe ; 0 et 20 > 15 mm, donc sans chaînage ils seraient séparés.
        let xs = [100.0_f32, 0.0, 105.0, 10.0, 20.0];
        let stacks: Vec<Stack> = xs.iter().enumerate().map(|(i, &x)| stack_masque_en(&t, &format!("s{i}"), x)).collect();
        let g = group_stacks(&stacks, 15.0).unwrap();
        assert_eq!(g.groups, vec![vec![1, 3, 4], vec![0, 2]], "plus grand groupe d'abord, indices croissants");
        // matrice de distances : symétrique, diagonale nulle, valeurs attendues
        assert_eq!(g.distances.len(), 5);
        for i in 0..5 {
            assert_eq!(g.distances[i][i], 0.0);
            for j in 0..5 {
                assert_eq!(g.distances[i][j], g.distances[j][i]);
            }
        }
        assert!((g.distances[1][3] - 10.0).abs() < 1e-9 && (g.distances[1][4] - 20.0).abs() < 1e-9);
        // seuil inclusif : à exactement 10 mm, les stacks 1 et 3 sont liés ; à 9,9 mm ils ne le sont plus
        assert_eq!(group_stacks(&stacks[1..2], 0.0).unwrap().groups, vec![vec![0]]);
        let a = group_stacks(&[stack_masque_en(&t, "a", 0.0), stack_masque_en(&t, "b", 10.0)], 10.0).unwrap();
        assert_eq!(a.groups, vec![vec![0, 1]]);
        let b = group_stacks(&[stack_masque_en(&t, "a", 0.0), stack_masque_en(&t, "b", 10.0)], 9.9).unwrap();
        assert_eq!(b.groups, vec![vec![0], vec![1]]);
        // à taille égale, le groupe de plus petit indice d'abord ; liste vide : aucun groupe
        let c = group_stacks(&[stack_masque_en(&t, "a", 50.0), stack_masque_en(&t, "b", 0.0)], 5.0).unwrap();
        assert_eq!(c.groups, vec![vec![0], vec![1]]);
        assert!(group_stacks(&[], 18.0).unwrap().groups.is_empty());
    }

    #[test]
    fn grouping_requires_a_brain_mask() {
        let t = Temp::new("groupes_sans_masque");
        let (f, _) = stack_et_masque(&t, &[[1, 1, 1]], None);
        let sans = Stack::read(&f).unwrap();
        let e = group_stacks(&[sans], 18.0).unwrap_err();
        assert!(matches!(e, SvrError::NoMask(_)), "{e}");
    }

    /// Compare, pour des sujets du jeu de développement, groupes et distances à la référence Python
    /// (`scripts/make_reference_groups.py`), sans nommer aucun sujet : `selection` choisit lesquels.
    fn verifier_groupes(mut selection: impl FnMut(usize, &str, usize) -> bool) {
        let racine = racine();
        let masques: std::collections::HashMap<String, String> =
            lire_tsv("data/reference/pivots/masques.tsv").into_iter().map(|c| (c[0].clone(), c[1].clone())).collect();
        let mut attendu: std::collections::BTreeMap<String, Vec<(usize, String, usize)>> = Default::default();
        for c in lire_tsv("data/reference/groups/groups.tsv") {
            attendu.entry(c[0].clone()).or_default().push((c[1].parse().unwrap(), c[2].clone(), c[3].parse().unwrap()));
        }
        let mut dist: std::collections::HashMap<(String, usize, usize), f64> = Default::default();
        for c in lire_tsv("data/reference/groups/distances.tsv") {
            dist.insert((c[0].clone(), c[1].parse().unwrap(), c[2].parse().unwrap()), c[3].parse().unwrap());
        }
        // Le fichier liste les stacks groupe par groupe : on les remet dans l'ordre des indices avant de les charger.
        for items in attendu.values_mut() {
            items.sort_by_key(|x| x.0);
        }
        let (mut n_sujets, mut multi, mut pire) = (0, 0, 0.0_f64);
        for (rang_sujet, (sujet, items)) in attendu.iter().enumerate() {
            let n_groupes_ref = items.iter().map(|x| x.2).max().unwrap() + 1;
            if !selection(rang_sujet, sujet, n_groupes_ref) {
                continue;
            }
            n_sujets += 1;
            let mut stacks = Vec::new();
            for (_, chemin, _) in items {
                let mut s = Stack::read(Path::new(&format!("{racine}/{chemin}"))).unwrap();
                s.set_brain_mask(Path::new(&format!("{racine}/{}", masques[chemin]))).unwrap();
                stacks.push(s);
            }
            let g = group_stacks(&stacks, DEFAULT_GROUP_GAP_MM).unwrap();
            // groupes de Rust = ceux de la référence (rang de groupe par indice de stack)
            let mut rang = vec![usize::MAX; stacks.len()];
            for (r, groupe) in g.groups.iter().enumerate() {
                for &i in groupe {
                    rang[i] = r;
                }
            }
            for (i, _, r) in items {
                assert_eq!(rang[*i], *r, "{sujet} : stack {i}");
            }
            multi += usize::from(g.groups.len() > 1);
            for i in 0..stacks.len() {
                for j in (i + 1)..stacks.len() {
                    pire = pire.max((g.distances[i][j] - dist[&(sujet.clone(), i, j)]).abs());
                }
            }
        }
        println!("{n_sujets} sujets, dont {multi} à plusieurs groupes ; écart max des distances à la référence {pire:.2e} mm");
        assert!(pire < 1e-6, "{pire:.2e} mm");
    }

    /// Version rapide : les sujets à plusieurs groupes et les deux premiers autres.
    #[test]
    fn stack_groups_match_python_reference_sampled() {
        let mut autres = 0;
        verifier_groupes(|_, _, n_groupes| {
            if n_groupes > 1 {
                return true;
            }
            autres += 1;
            autres <= 2
        });
    }

    /// Version complète : les 15 sujets (plus d'une minute en debug). Lancer avec :
    /// `cargo test -p medoxide-svr -- --ignored --nocapture stack_groups_match_python_reference_all`
    #[test]
    #[ignore = "long : voir la doc du test"]
    fn stack_groups_match_python_reference_all() {
        verifier_groupes(|_, _, _| true);
    }
}
