//! Critère 1 de P4 bis (docs/ULF.md) : INR à grille de hachage comparée au code PyTorch de fmi, mêmes poids.
//!
//! Cas synthétique versionné (`tests/donnees/inr_synthetique.h5`, écrit par `scripts/cas_inr.py synthetique`) ; réglages
//! de P1 sur 115 × 115 × 45 si `MEDOXIDE_ULF_INR` désigne le fichier écrit par `scripts/cas_inr.py p1` (hors dépôt pour
//! sa taille, environ 1,4 Go ; aucune donnée de sujet).
//!
//! Seuils : indices de table identiques, coordonnées bit pour bit ; sorties à ≤ 1e-12 (`f64`) et ≤ 1e-6 (`f32`),
//! gradients de L = Σ r · sortie à ≤ 1e-10 et ≤ 1e-5 (écarts relatifs L2). Amendement du 2026-11-22 (docs/ULF.md) : en
//! `f32`, une quantité passe aussi si elle reste sous deux fois l'écart de PyTorch float32 au float64 sur le même cas
//! (conditionnement : centrage du champ, sommes sur 595 125 points). `MEDOXIDE_ULF_GPU` ajoute la vérification `f32` sur
//! wgpu, `MEDOXIDE_ULF_CUDA` (avec la feature `cuda`) sur CUDA.

use std::path::Path;

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use medoxide_ulf::inr::{indices_et_poids, linspace, ChampB0, ConfigGrille, Encodage, ImageInr, InrHachage};

fn lire(f: &hdf5::Group, nom: &str) -> (Vec<f64>, Vec<usize>) {
    let d = f.dataset(nom).unwrap();
    (d.read_raw::<f64>().unwrap(), d.shape())
}

fn attr_f64(g: &hdf5::Group, nom: &str) -> f64 {
    g.attr(nom).unwrap().read_scalar::<f64>().unwrap()
}

fn attr_i64(g: &hdf5::Group, nom: &str) -> i64 {
    g.attr(nom).unwrap().read_scalar::<i64>().unwrap()
}

fn en_f64<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.try_into_data_dtype(DType::F64).unwrap().try_to_vec::<f64>().unwrap()
}

/// Écart relatif L2 de `a` à la référence `r`.
fn ecart(a: &[f64], r: &[f64]) -> f64 {
    assert_eq!(a.len(), r.len());
    let num: f64 = a.iter().zip(r).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = r.iter().map(|y| y * y).sum();
    (num / den).sqrt()
}

fn config(g: &hdf5::Group) -> ConfigGrille {
    ConfigGrille {
        n_niveaux: attr_i64(g, "n_levels") as usize,
        n_caracteristiques: attr_i64(g, "n_features_per_level") as usize,
        log2_table: attr_i64(g, "log2_hashmap_size") as u32,
        resolution_base: attr_f64(g, "base_resolution"),
        facteur: attr_f64(g, "per_level_scale"),
    }
}

fn reseau(g: &hdf5::Group, cfg: &ConfigGrille, device: &Device, dtype: DType) -> InrHachage {
    let tables = (0..cfg.n_niveaux)
        .map(|l| {
            let (v, f) = lire(g, &format!("niveau_{l}/table"));
            (v, [f[0], f[1]])
        })
        .collect();
    let n_couches = g
        .member_names()
        .unwrap()
        .iter()
        .filter(|n| n.starts_with("couche_"))
        .count();
    let couches = (0..n_couches)
        .map(|c| {
            let (w, f) = lire(g, &format!("couche_{c}/poids"));
            (w, [f[0], f[1]], lire(g, &format!("couche_{c}/biais")).0)
        })
        .collect();
    InrHachage::depuis_poids(tables, couches, device, dtype)
}

/// Écarts relevés et quantités hors seuil. En `f64` : seuils du critère. En `f32` (amendement du 2026-11-22) : seuil du
/// critère ou deux fois l'écart de PyTorch float32 au float64 sur le même cas (`ecart_pytorch_f32/…`), le plus grand.
#[derive(Debug)]
struct Ecarts {
    f32: bool,
    sorties: f64,
    gradients: f64,
    echecs: Vec<String>,
}

impl Ecarts {
    fn nouveau(dtype: DType) -> Self {
        Self {
            f32: dtype == DType::F32,
            sorties: 0.0,
            gradients: 0.0,
            echecs: Vec::new(),
        }
    }

    fn verifier(&mut self, nom: &str, e: f64, gradient: bool, plancher: Option<f64>) {
        let seuil: f64 = match (self.f32, gradient) {
            (false, false) => 1e-12,
            (false, true) => 1e-10,
            (true, false) => 1e-6,
            (true, true) => 1e-5,
        };
        let limite = if self.f32 {
            plancher.map_or(seuil, |p| seuil.max(2.0 * p))
        } else {
            seuil
        };
        let pytorch = plancher
            .filter(|_| self.f32)
            .map_or(String::new(), |p| format!("  (PyTorch f32 {p:.2e})"));
        println!("  {}{nom:<28} {e:.2e}{pytorch}", if gradient { "grad " } else { "" });
        if gradient {
            self.gradients = self.gradients.max(e);
        } else {
            self.sorties = self.sorties.max(e);
        }
        if e > limite || e.is_nan() {
            self.echecs.push(format!("{nom} : {e:.2e} > {limite:.2e}"));
        }
    }
}

/// Géométrie des niveaux et indices de table, entier pour entier ; coordonnées bit pour bit.
fn verifier_indices(f: &hdf5::File, dims: [usize; 3]) {
    let (coords, _) = lire(f, "coordonnees");
    let axes: Vec<Vec<f64>> = dims.iter().map(|&n| linspace(n)).collect();
    for (p, c) in coords.as_chunks::<3>().0.iter().enumerate() {
        let i = [p / (dims[1] * dims[2]), (p / dims[2]) % dims[1], p % dims[2]];
        for d in 0..3 {
            assert_eq!(c[d].to_bits(), axes[d][i[d]].to_bits(), "coordonnée {p}, axe {d}");
        }
    }
    for nom in ["champ", "image"] {
        let g = f.group(nom).unwrap();
        let cfg = config(&g);
        for l in 0..cfg.n_niveaux {
            let gl = g.group(&format!("niveau_{l}")).unwrap();
            let niveau = cfg.niveau(l);
            assert_eq!(
                niveau.echelle.to_bits(),
                attr_f64(&gl, "echelle").to_bits(),
                "{nom} {l} échelle"
            );
            assert_eq!(niveau.resolution, attr_i64(&gl, "resolution"), "{nom} {l} résolution");
            assert_eq!(
                niveau.taille_table as i64,
                attr_i64(&gl, "taille_table"),
                "{nom} {l} table"
            );
            assert_eq!(
                niveau.hachee,
                gl.attr("hachee").unwrap().read_scalar::<bool>().unwrap(),
                "{nom} {l} hachage"
            );
            let attendus = gl.dataset("indices").unwrap().read_raw::<i64>().unwrap();
            let (indices, _) = indices_et_poids(&niveau, dims);
            let differents = indices.iter().zip(&attendus).filter(|(a, b)| a != b).count();
            assert_eq!(indices.len(), attendus.len());
            assert_eq!(
                differents, 0,
                "{nom} niveau {l} : {differents} indices différents de PyTorch"
            );
        }
    }
}

/// Sorties et gradients du champ et de l'image dans le type `dtype`.
fn verifier_reseaux(f: &hdf5::File, dims: [usize; 3], dtype: DType, device: &Device) -> Ecarts {
    let device = device.clone();
    let opts = (&device, dtype);
    let mut e = Ecarts::nouveau(dtype);
    for nom in ["champ", "image"] {
        println!("{nom} ({dtype:?}, {device:?})");
        let g = f.group(nom).unwrap();
        let plancher = |cle: &str| {
            g.dataset(&format!("ecart_pytorch_f32/{cle}"))
                .ok()
                .map(|d| d.read_scalar::<f64>().unwrap())
        };
        let cfg = config(&g);
        let encodage = Encodage::nouveau(&cfg, dims, &device, dtype);
        let net = reseau(&g, &cfg, &device, dtype);
        for l in 0..cfg.n_niveaux {
            let enc = en_f64(net.encodage_niveau(&encodage, l));
            e.verifier(
                &format!("encodage niveau {l}"),
                ecart(&enc, &lire(&g, &format!("niveau_{l}/encodage")).0),
                false,
                None,
            );
        }
        let brute = ecart(&en_f64(net.evaluer(&encodage)), &lire(&g, "sortie").0);
        e.verifier("sortie brute", brute, false, plancher("sortie"));
        let r = lire(&g, "r").0;
        let n = encodage.n_points();
        let (perte, net) = if nom == "champ" {
            let mut champ = ChampB0::nouveau(net, attr_f64(&g, "b0_min_hz"), attr_f64(&g, "b0_max_hz"));
            let scalaire = |v: f64| Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![v], [1]), opts));
            champ.s = scalaire(lire(&g, "s").0[0]);
            champ.o = scalaire(lire(&g, "o").0[0]);
            let b0 = champ.b0(&encodage);
            e.verifier(
                "b0",
                ecart(&en_f64(b0.clone()), &lire(&g, "b0").0),
                false,
                plancher("b0"),
            );
            let perte = (b0 * Tensor::from_data(TensorData::new(r, [n]), opts)).sum();
            let grads = perte.backward();
            for (p, cle) in [(&champ.s, "s"), (&champ.o, "o")] {
                let cle = format!("grad/{cle}");
                let ecart = ecart(&en_f64(p.val().grad(&grads).unwrap()), &lire(&g, &cle).0);
                e.verifier(&cle[5..], ecart, true, plancher(&cle));
            }
            (grads, champ.reseau)
        } else {
            let image = ImageInr { reseau: net };
            let (re, im) = image.image(&encodage);
            e.verifier(
                "re",
                ecart(&en_f64(re.clone()), &lire(&g, "re").0),
                false,
                plancher("re"),
            );
            e.verifier(
                "im",
                ecart(&en_f64(im.clone()), &lire(&g, "im").0),
                false,
                plancher("im"),
            );
            let r = Tensor::<2>::from_data(TensorData::new(r, [2, n]), opts);
            let perte = (re * r.clone().slice([0..1, 0..n]).reshape([n])).sum()
                + (im * r.slice([1..2, 0..n]).reshape([n])).sum();
            (perte.backward(), image.reseau)
        };
        let grads = perte;
        let mut gradient = |cle: String, t: Tensor<2>| {
            let ecart = ecart(&en_f64(t), &lire(&g, &format!("grad/{cle}")).0);
            e.verifier(&cle, ecart, true, plancher(&format!("grad/{cle}")));
        };
        for (l, t) in net.tables.iter().enumerate() {
            gradient(format!("niveau_{l}/table"), t.val().grad(&grads).unwrap());
        }
        for (c, couche) in net.couches.iter().enumerate() {
            // Burn range les poids en (entrées, sorties), PyTorch en (sorties, entrées).
            gradient(
                format!("couche_{c}/poids"),
                couche.weight.val().grad(&grads).unwrap().transpose(),
            );
            let gb = couche.bias.as_ref().unwrap().val().grad(&grads).unwrap();
            gradient(format!("couche_{c}/biais"), gb.unsqueeze::<2>());
        }
    }
    e
}

fn verifier(chemin: &Path) {
    let f = hdf5::File::open(chemin).unwrap();
    let d = f.attr("dims").unwrap().read_raw::<i64>().unwrap();
    let dims = [d[0] as usize, d[1] as usize, d[2] as usize];
    verifier_indices(&f, dims);
    let cpu = Device::flex().autodiff();
    let mut ecarts = vec![("f64 CPU", verifier_reseaux(&f, dims, DType::F64, &cpu))];
    ecarts.push(("f32 CPU", verifier_reseaux(&f, dims, DType::F32, &cpu)));
    // GPU (wgpu : Metal, Vulkan) si MEDOXIDE_ULF_GPU est défini : c'est la cible des calculs en f32.
    if std::env::var("MEDOXIDE_ULF_GPU").is_ok() {
        let gpu = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff();
        ecarts.push(("f32 wgpu", verifier_reseaux(&f, dims, DType::F32, &gpu)));
    }
    // CUDA (feature `cuda`) si MEDOXIDE_ULF_CUDA est défini : la cible des calculs sur la RTX.
    #[cfg(feature = "cuda")]
    if std::env::var("MEDOXIDE_ULF_CUDA").is_ok() {
        let gpu = Device::cuda(0).autodiff();
        ecarts.push(("f32 cuda", verifier_reseaux(&f, dims, DType::F32, &gpu)));
    }
    for (nom, e) in &ecarts {
        println!(
            "{nom} : plus grands écarts, sorties {:.2e}, gradients {:.2e}",
            e.sorties, e.gradients
        );
    }
    let echecs: Vec<String> = ecarts
        .iter()
        .flat_map(|(nom, e)| e.echecs.iter().map(move |x| format!("{nom} {x}")))
        .collect();
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}

#[test]
fn inr_comme_pytorch_cas_synthetique() {
    verifier(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/inr_synthetique.h5"));
}

#[test]
fn inr_comme_pytorch_reglages_p1() {
    let Ok(chemin) = std::env::var("MEDOXIDE_ULF_INR") else {
        eprintln!("MEDOXIDE_ULF_INR non défini : cas aux réglages de P1 sauté");
        return;
    };
    verifier(Path::new(&chemin));
}

/// Initialisation de fmi : tables dans ± 1e-4, couches dans ± 1/√(entrées), scalaires du champ nuls, type respecté.
#[test]
fn initialisation_comme_fmi() {
    let device = Device::flex();
    for dtype in [DType::F32, DType::F64] {
        let cfg = ConfigGrille::champ_p1();
        let champ = ChampB0::nouveau(InrHachage::nouveau(&cfg, 1, 64, 2, &device, dtype), 12.5, 108.0);
        assert_eq!(champ.reseau.tables.len(), 8);
        for (l, t) in champ.reseau.tables.iter().enumerate() {
            assert_eq!(t.val().dims(), [cfg.niveau(l).taille_table, 2]);
            assert_eq!(t.val().dtype(), dtype);
            let v = en_f64(t.val());
            assert!(v.iter().all(|x| x.abs() <= 1e-4) && v.iter().any(|x| x.abs() > 5e-5));
        }
        let entrees = [16, 64, 64];
        for (couche, n) in champ.reseau.couches.iter().zip(entrees) {
            let borne = 1.0 / (n as f64).sqrt();
            let poids = en_f64(couche.weight.val());
            assert!(poids.iter().all(|x| x.abs() <= borne) && poids.iter().any(|x| x.abs() > borne / 2.0));
            assert!(en_f64(couche.bias.as_ref().unwrap().val())
                .iter()
                .all(|x| x.abs() <= borne));
        }
        assert_eq!(champ.reseau.couches[2].weight.val().dims(), [64, 1]);
        assert_eq!((en_f64(champ.s.val()), en_f64(champ.o.val())), (vec![0.0], vec![0.0]));
        // s = o = 0 : amplitude = moyenne géométrique des bornes, offset nul.
        let a = en_f64(champ.amplitude_hz())[0];
        assert!((a - (12.5f64 * 108.0).sqrt()).abs() < 1e-3 * a, "{a}");
    }
}
