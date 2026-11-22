//! Critère 4 de P4 bis (docs/ULF.md) : pas d'optimisation de l'estimation jointe comparés à PyTorch (fmi), à partir des
//! mêmes poids initiaux, en `f64`.
//!
//! Cas synthétique versionné (`tests/donnees/jointe_synthetique.h5`, écrit par `scripts/cas_jointe.py synthetique`) ;
//! plans de P1 si `MEDOXIDE_ULF_P1_JOINTE` désigne un dossier de fichiers écrits par `scripts/cas_jointe.py p1` (hors
//! dépôt). Deux pas de démarrage à chaud par réseau depuis les poids initiaux ; deux pas joints depuis les poids de
//! PyTorch après le démarrage complet (300 pas par réseau, comme dans la méthode). Deux pas plutôt qu'un, pour exercer
//! les corrections de biais d'Adam et la décroissance cosinus au-delà du premier pas.
//!
//! Seuil : après chaque phase, chaque paramètre (table, poids, biais, scalaire) à ≤ 1e-10 de PyTorch en écart relatif
//! L2. Amendement du 2026-11-22 (docs/ULF.md) : sur les deux démarrages, un paramètre passe aussi s'il reste sous deux
//! fois l'écart de PyTorch à lui-même entre ses calculs à un thread et au nombre de threads par défaut
//! (`plancher_pytorch/…`, mesuré par le script) : à l'initialisation, la sortie du réseau du champ est presque constante
//! et son centrage perd des chiffres. Sont aussi vérifiés la cible du démarrage de l'image (≤ 1e-12) et les termes de la
//! perte à chaque pas joint (≤ 1e-10), et affiché l'écart rapporté au déplacement de PyTorch depuis l'état précédent.

use std::path::Path;

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use medoxide_ulf::inr::{ChampB0, ConfigGrille, Encodage, ImageInr, InrHachage};
use medoxide_ulf::jointe::{pas_joint, pas_prefit_champ, pas_prefit_image, Adam, CibleChamp, CibleImage, Reglages};
use medoxide_ulf::pertes::Donnees;
use ndarray::{Array1, Array3, Ix1, Ix3};
use rustfft::num_complex::Complex64;

const SEUIL: f64 = 1e-10;

fn lire(g: &hdf5::Group, nom: &str) -> (Vec<f64>, Vec<usize>) {
    let d = g.dataset(nom).unwrap();
    (d.read_raw::<f64>().unwrap(), d.shape())
}

fn n_membres(g: &hdf5::Group, prefixe: &str) -> usize {
    g.member_names()
        .unwrap()
        .iter()
        .filter(|n| n.starts_with(prefixe))
        .count()
}

fn config(g: &hdf5::Group) -> ConfigGrille {
    let f = |n: &str| g.attr(n).unwrap().read_scalar::<f64>().unwrap();
    let i = |n: &str| g.attr(n).unwrap().read_scalar::<i64>().unwrap();
    ConfigGrille {
        n_niveaux: i("n_levels") as usize,
        n_caracteristiques: i("n_features_per_level") as usize,
        log2_table: i("log2_hashmap_size") as u32,
        resolution_base: f("base_resolution"),
        facteur: f("per_level_scale"),
    }
}

/// Paramètres d'un réseau au format PyTorch : (nom, valeurs) ; poids de forme (sorties, entrées).
fn parametres_h5(g: &hdf5::Group) -> Vec<(String, Vec<f64>, Vec<usize>)> {
    let mut p = Vec::new();
    for l in 0..n_membres(g, "niveau_") {
        let nom = format!("niveau_{l}/table");
        let (v, f) = lire(g, &nom);
        p.push((nom, v, f));
    }
    for c in 0..n_membres(g, "couche_") {
        for q in ["poids", "biais"] {
            let nom = format!("couche_{c}/{q}");
            let (v, f) = lire(g, &nom);
            p.push((nom, v, f));
        }
    }
    p
}

fn reseau(g: &hdf5::Group, device: &Device) -> InrHachage {
    let p = parametres_h5(g);
    let n_niveaux = n_membres(g, "niveau_");
    let tables = p[..n_niveaux]
        .iter()
        .map(|(_, v, f)| (v.clone(), [f[0], f[1]]))
        .collect();
    let couches = p[n_niveaux..]
        .chunks(2)
        .map(|c| (c[0].1.clone(), [c[0].2[0], c[0].2[1]], c[1].1.clone()))
        .collect();
    InrHachage::depuis_poids(tables, couches, device, DType::F64)
}

fn scalaire_h5(g: &hdf5::Group, nom: &str) -> f64 {
    g.dataset(nom).unwrap().read_scalar::<f64>().unwrap()
}

fn param_scalaire(v: f64, device: &Device) -> Param<Tensor<1>> {
    Param::from_tensor(Tensor::from_data(TensorData::new(vec![v], [1]), (device, DType::F64)))
}

fn en_f64<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.try_into_data_dtype(DType::F64).unwrap().try_to_vec::<f64>().unwrap()
}

/// Paramètres d'un réseau de Rust, dans l'ordre et au format de [`parametres_h5`].
fn parametres_rust(r: &InrHachage) -> Vec<Vec<f64>> {
    let mut p: Vec<Vec<f64>> = r.tables.iter().map(|t| en_f64(t.val())).collect();
    for c in &r.couches {
        p.push(en_f64(c.weight.val().transpose()));
        p.push(en_f64(c.bias.as_ref().unwrap().val()));
    }
    p
}

fn noter(echecs: &mut Vec<String>, nom: String, e: f64, seuil: f64) {
    if e > seuil || e.is_nan() {
        echecs.push(format!("{nom} : {e:.2e} > {seuil:.2e}"));
    }
}

fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0, f64::max)
}

fn norme(v: impl Iterator<Item = f64>) -> f64 {
    v.map(|x| x * x).sum::<f64>().sqrt()
}

/// Compare des paramètres à la référence ; `avant` : référence à l'état précédent, pour l'écart rapporté au
/// déplacement (affiché). Rend l'écart relatif de chaque paramètre.
fn comparer(nom: &str, rust: &[Vec<f64>], reference: &[Vec<f64>], avant: &[Vec<f64>]) -> Vec<f64> {
    let mut d_max = 0.0f64;
    let ecarts: Vec<f64> = rust
        .iter()
        .zip(reference)
        .zip(avant)
        .map(|((a, r), p)| {
            assert_eq!(a.len(), r.len());
            let diff = norme(a.iter().zip(r).map(|(x, y)| x - y));
            let d = diff / norme(r.iter().zip(p).map(|(x, y)| x - y));
            if d.is_finite() {
                d_max = d_max.max(d);
            }
            let e = diff / norme(r.iter().copied());
            if e.is_nan() {
                f64::INFINITY
            } else {
                e
            }
        })
        .collect();
    let e_max = ecarts.iter().copied().fold(0.0, f64::max);
    println!("  {nom:<32} écart relatif {e_max:.2e}   rapporté au déplacement {d_max:.2e}");
    ecarts
}

/// Noms des paramètres d'un réseau du fichier (avec `s` et `o` pour le champ), dans l'ordre de [`parametres_h5`].
fn noms(g: &hdf5::Group, champ: bool) -> Vec<String> {
    let mut n: Vec<String> = parametres_h5(g).into_iter().map(|(nom, _, _)| nom).collect();
    if champ {
        n.extend(["s".to_string(), "o".to_string()]);
    }
    n
}

fn parametres_champ_h5(g: &hdf5::Group) -> Vec<Vec<f64>> {
    let mut p: Vec<Vec<f64>> = parametres_h5(g).into_iter().map(|(_, v, _)| v).collect();
    p.push(vec![scalaire_h5(g, "s")]);
    p.push(vec![scalaire_h5(g, "o")]);
    p
}

fn parametres_champ(c: &ChampB0) -> Vec<Vec<f64>> {
    let mut p = parametres_rust(&c.reseau);
    p.push(en_f64(c.s.val()));
    p.push(en_f64(c.o.val()));
    p
}

fn parametres_image_h5(g: &hdf5::Group) -> Vec<Vec<f64>> {
    parametres_h5(g).into_iter().map(|(_, v, _)| v).collect()
}

/// Rejoue les pas du fichier ; rend les échecs.
fn verifier(chemin: &Path) -> Vec<String> {
    println!("{}", chemin.display());
    let f = hdf5::File::open(chemin).unwrap();
    let device = Device::flex().autodiff();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    let (k_reg, k_inv) = (c("k_reg"), c("k_inv"));
    let temps = f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap();
    let e = f.dataset("espacement_mm").unwrap().read_raw::<f64>().unwrap();
    let b0_init: Array3<f64> = f.dataset("b0_init").unwrap().read::<f64, Ix3>().unwrap();
    let bornes: Array1<f64> = f.dataset("bornes_b0").unwrap().read::<f64, Ix1>().unwrap();
    let attr = |n: &str| f.attr(n).unwrap().read_scalar::<i64>().unwrap() as usize;
    let (n_prefit, n_joint) = (attr("n_prefit"), attr("n_joint"));
    let reglages = Reglages {
        n_iter: attr("n_iter_joint"),
        ..Reglages::default()
    };
    let d = Donnees::<f64>::nouveau(&k_reg, &k_inv, temps.as_slice().unwrap(), [e[0], e[1], e[2]], &device);
    let dims = d.dims;
    let enc_champ = Encodage::nouveau(&config(&f.group("champ_config").unwrap()), dims, &device, DType::F64);
    let enc_image = Encodage::nouveau(&config(&f.group("image_config").unwrap()), dims, &device, DType::F64);
    let groupe = |e: usize, nom: &str| f.group(&format!("etat_{e}/{nom}")).unwrap();

    let champ_h5 = |e: usize| {
        let g = groupe(e, "champ");
        ChampB0 {
            reseau: reseau(&g, &device),
            s: param_scalaire(scalaire_h5(&g, "s"), &device),
            o: param_scalaire(scalaire_h5(&g, "o"), &device),
            b0_min_hz: bornes[0],
            b0_max_hz: bornes[1],
        }
    };
    let image_h5 = |e: usize| ImageInr {
        reseau: reseau(&groupe(e, "image"), &device),
    };
    let g0 = groupe(0, "champ");
    let mut champ = champ_h5(0);
    let mut image = image_h5(0);
    let mut echecs = Vec::new();
    // Démarrages (amendement du 2026-11-22) : chaque paramètre sous SEUIL ou sous deux fois le plancher de PyTorch
    // (écart entre ses calculs à un thread et au nombre de threads par défaut).
    let plancher = |e: usize, reseau: &str, p: &str| {
        f.dataset(&format!("plancher_pytorch/etat_{e}/{reseau}/{p}"))
            .unwrap()
            .read_scalar::<f64>()
            .unwrap()
    };
    let verifier_demarrage = |echecs: &mut Vec<String>, e: usize, reseau: &str, ecarts: &[f64], noms: &[String]| {
        let mut pire = (0.0f64, 0.0f64, String::new());
        for (ecart, p) in ecarts.iter().zip(noms) {
            let pl = plancher(e, reseau, p);
            let limite = SEUIL.max(2.0 * pl);
            if ecart / limite > pire.0 / pire.1.max(f64::MIN_POSITIVE) || pire.2.is_empty() {
                pire = (*ecart, limite, p.clone());
            }
            noter(
                echecs,
                format!("démarrage, {reseau}, {p} (PyTorch contre lui-même {pl:.2e})"),
                *ecart,
                limite,
            );
        }
        println!(
            "  {:<32} au plus près de sa limite : {} à {:.2e} pour {:.2e}",
            "", pire.2, pire.0, pire.1
        );
    };

    // démarrage à chaud du champ
    let cible = CibleChamp::nouvelle(&b0_init, &device, DType::F64);
    let mut adam = Adam::default();
    for _ in 0..n_prefit {
        champ = pas_prefit_champ(champ, &enc_champ, &cible, &mut adam, reglages.lr_prefit).0;
    }
    let r = parametres_champ_h5(&groupe(1, "champ"));
    let e1 = comparer(
        "démarrage du champ : champ",
        &parametres_champ(&champ),
        &r,
        &parametres_champ_h5(&g0),
    );
    verifier_demarrage(&mut echecs, 1, "champ", &e1, &noms(&g0, true));

    // démarrage à chaud de l'image
    let cible = CibleImage::nouvelle(&d, &b0_init, &device, DType::F64);
    let ref_cible: Array3<f64> = f.dataset("cible_image").unwrap().read::<f64, Ix3>().unwrap();
    let e_cible = norme(cible.magnitude.iter().zip(&ref_cible).map(|(a, b)| a - b)) / norme(ref_cible.iter().copied());
    println!("  cible du démarrage de l'image       écart relatif {e_cible:.2e}");
    noter(&mut echecs, "cible de l'image".into(), e_cible, 1e-12);
    let mut adam = Adam::default();
    for _ in 0..n_prefit {
        image = pas_prefit_image(image, &enc_image, &cible, &mut adam, reglages.lr_prefit).0;
    }
    let r = parametres_image_h5(&groupe(2, "image"));
    let avant = parametres_image_h5(&groupe(0, "image"));
    let e2 = comparer(
        "démarrage de l'image : image",
        &parametres_rust(&image.reseau),
        &r,
        &avant,
    );
    verifier_demarrage(&mut echecs, 2, "image", &e2, &noms(&groupe(0, "image"), false));

    // boucle jointe, depuis les poids de PyTorch après le démarrage complet (état 3)
    let (mut champ, mut image) = (champ_h5(3), image_h5(3));
    let mut ref_champ = parametres_champ_h5(&groupe(3, "champ"));
    let mut ref_image = parametres_image_h5(&groupe(3, "image"));
    let pertes = |nom: &str| {
        f.dataset(&format!("pertes_jointes/{nom}"))
            .unwrap()
            .read::<f64, Ix1>()
            .unwrap()
    };
    let mut adam = Adam::default();
    for it in 0..n_joint {
        let (c, i, bilan) = pas_joint(
            &d,
            champ,
            image,
            &enc_champ,
            &enc_image,
            &mut adam,
            &reglages,
            reglages.facteur_lr(it),
        );
        (champ, image) = (c, i);
        for (nom, v) in [
            ("fidelity", bilan.fidelite),
            ("image", bilan.image),
            ("b0", bilan.b0),
            ("smooth", bilan.lissage),
            ("support", bilan.support),
            ("totale", bilan.totale),
        ] {
            let r = pertes(nom)[it];
            let e = ((v - r) / r).abs();
            println!("  pas joint {it}, terme {nom:<9} {v:+.12e} (PyTorch {r:+.12e})  écart {e:.2e}");
            noter(&mut echecs, format!("pas joint {it}, terme {nom}"), e, SEUIL);
        }
        let rc = parametres_champ_h5(&groupe(4 + it, "champ"));
        let ri = parametres_image_h5(&groupe(4 + it, "image"));
        let ec = max(&comparer(
            &format!("pas joint {it} : champ"),
            &parametres_champ(&champ),
            &rc,
            &ref_champ,
        ));
        let ei = max(&comparer(
            &format!("pas joint {it} : image"),
            &parametres_rust(&image.reseau),
            &ri,
            &ref_image,
        ));
        noter(&mut echecs, format!("pas joint {it}, champ"), ec, SEUIL);
        noter(&mut echecs, format!("pas joint {it}, image"), ei, SEUIL);
        (ref_champ, ref_image) = (rc, ri);
    }
    echecs
}

#[test]
fn pas_comme_pytorch_cas_synthetique() {
    let echecs = verifier(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/jointe_synthetique.h5"));
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}

#[test]
fn pas_comme_pytorch_plans_p1() {
    let Ok(dossier) = std::env::var("MEDOXIDE_ULF_P1_JOINTE") else {
        eprintln!("MEDOXIDE_ULF_P1_JOINTE absent : test sur les plans de P1 sauté");
        return;
    };
    let mut fichiers: Vec<_> = std::fs::read_dir(&dossier)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "h5"))
        .collect();
    fichiers.sort();
    assert!(!fichiers.is_empty(), "{dossier} : aucun fichier .h5");
    let echecs: Vec<String> = fichiers
        .iter()
        .flat_map(|c| verifier(c).into_iter().map(move |e| format!("{} : {e}", c.display())))
        .collect();
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}
