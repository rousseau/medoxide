//! Critères 1 à 3 et 5 de P6 (docs/ULF.md) : entrées de l'étape 1 depuis une paire de fichiers ISMRMRD, comparées au
//! code de fmi (`utils_io.load_pair`, `denoise_joint.Context`).
//!
//! Cas synthétique versionné (`tests/donnees/entrees_synthetique/`, écrit par `scripts/cas_entrees.py synthetique`) ;
//! plans de P1 si `MEDOXIDE_ULF_P1_ENTREES` désigne un dossier de `<sujet>_<plan>.h5` écrits par `scripts/cas_entrees.py
//! reel` (valeurs des `inputs.npz` de P1), avec les fichiers d'acquisition dans son sous-dossier `fichiers/` (hors dépôt).
//!
//! Seuils : positions d'écho et demi-largeur à ≤ 1e-9 échantillon ; k-space fenêtré à ≤ 1e-12 (écart relatif L2) ;
//! temps, bornes et espacement à ≤ 1e-12 (relatif) ; masque identique, sauf voxels à moins de 1e-9 (relatif) du seuil,
//! rapportés.

use std::path::{Path, PathBuf};

use medoxide_ulf::entrees::{depuis_fichiers, image_masque};
use ndarray::{Array3, Ix1, Ix3};
use rustfft::num_complex::Complex64;

fn ecart_l2(a: &Array3<Complex64>, r: &Array3<Complex64>) -> f64 {
    let num: f64 = a.iter().zip(r).map(|(x, y)| (x - y).norm_sqr()).sum();
    let den: f64 = r.iter().map(|y| y.norm_sqr()).sum();
    (num / den).sqrt()
}

fn ecart_rel(a: &[f64], r: &[f64]) -> f64 {
    assert_eq!(a.len(), r.len());
    a.iter()
        .zip(r)
        .map(|(x, y)| (x - y).abs() / y.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, |m, e| if e.is_nan() || e > m { e } else { m }) // NaN propagé
}

fn verifier(attendu: &Path, fichiers: &Path) -> Vec<String> {
    let nom = attendu.file_stem().unwrap().to_string_lossy().to_string();
    let f = hdf5::File::open(attendu).unwrap();
    let attr_s = |n: &str| {
        f.attr(n)
            .unwrap()
            .read_scalar::<hdf5::types::VarLenUnicode>()
            .unwrap()
            .to_string()
    };
    let attr_f = |n: &str| f.attr(n).unwrap().read_scalar::<f64>().unwrap();
    let e = depuis_fichiers(
        &fichiers.join(attr_s("fichier_reg")),
        &fichiers.join(attr_s("fichier_inv")),
    )
    .unwrap();
    let mut echecs = Vec::new();
    let mut noter = |quoi: &str, v: f64, seuil: f64| {
        println!("  {nom} {quoi:<22} {v:.2e}");
        if v > seuil || v.is_nan() {
            echecs.push(format!("{nom} {quoi} : {v:.2e} > {seuil:.0e}"));
        }
    };
    noter("écho reg", (e.echo_reg - attr_f("echo_reg")).abs(), 1e-9);
    noter("écho inv", (e.echo_inv - attr_f("echo_inv")).abs(), 1e-9);
    noter("demi-largeur", (e.demi_largeur - attr_f("demi_largeur")).abs(), 1e-9);
    let c = |n: &str| f.dataset(n).unwrap().read::<Complex64, Ix3>().unwrap();
    noter("k_reg fenêtré", ecart_l2(&e.k_reg, &c("k_reg")), 1e-12);
    noter("k_inv fenêtré", ecart_l2(&e.k_inv, &c("k_inv")), 1e-12);
    let v = |n: &str| f.dataset(n).unwrap().read::<f64, Ix1>().unwrap().to_vec();
    noter("temps", ecart_rel(&e.temps, &v("temps")), 1e-12);
    noter("bornes", ecart_rel(&e.bornes_b0, &v("bornes_b0")), 1e-12);
    noter("espacement", ecart_rel(&e.espacement_mm, &v("espacement_mm")), 1e-12);

    let masque = f.dataset("masque").unwrap().read::<u8, Ix3>().unwrap();
    let (image, seuil) = image_masque(&e.k_reg, &e.k_inv, &e.temps);
    let (mut toleres, mut faux) = (0, 0);
    for ((m, r), x) in e.masque.iter().zip(&masque).zip(&image) {
        if *m != (*r != 0) {
            if ((x - seuil) / seuil).abs() <= 1e-9 {
                toleres += 1;
            } else {
                faux += 1;
            }
        }
    }
    println!(
        "  {nom} masque : {} voxels, {toleres} différents tolérés (au seuil), {faux} différents",
        masque.iter().filter(|&&r| r != 0).count()
    );
    if faux > 0 {
        echecs.push(format!("{nom} masque : {faux} voxels différents hors du seuil"));
    }
    echecs
}

#[test]
fn entrees_comme_fmi_cas_synthetique() {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/entrees_synthetique");
    let echecs = verifier(&d.join("attendu.h5"), &d);
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}

#[test]
fn entrees_comme_fmi_plans_p1() {
    let Ok(dossier) = std::env::var("MEDOXIDE_ULF_P1_ENTREES") else {
        eprintln!("MEDOXIDE_ULF_P1_ENTREES non défini : plans de P1 sautés");
        return;
    };
    let dossier = PathBuf::from(dossier);
    let mut attendus: Vec<_> = std::fs::read_dir(&dossier)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "h5"))
        .collect();
    attendus.sort();
    assert!(!attendus.is_empty());
    let echecs: Vec<String> = attendus
        .iter()
        .flat_map(|p| verifier(p, &dossier.join("fichiers")))
        .collect();
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}

#[test]
fn paire_incoherente_refusee() {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/entrees_synthetique");
    let refus = |reg: &Path, inv: &Path| depuis_fichiers(reg, inv).unwrap_err().to_string();
    // deux fois reg : polarités non opposées
    let e = refus(&d.join("reg.h5"), &d.join("reg.h5"));
    assert!(e.contains("polarités"), "{e}");
    // reg et inv échangés
    let e = refus(&d.join("inv.h5"), &d.join("reg.h5"));
    assert!(e.contains("polarités"), "{e}");
    // inv dont l'écrivain a aussi retourné les axes de phase (invert_ph_polarity), reg non
    let inv = medoxide_ismrmrd::lire(&d.join("inv.h5")).unwrap();
    let ligne = "<userParameterString><name>invert_ro_polarity</name><value>True</value></userParameterString>";
    assert!(inv.xml.contains(ligne));
    let xml: hdf5::types::VarLenUnicode = inv
        .xml
        .replace(
            ligne,
            &format!(
                "{ligne}<userParameterString><name>invert_ph_polarity</name><value>True</value></userParameterString>"
            ),
        )
        .parse()
        .unwrap();
    let dossier = std::env::temp_dir().join(format!("medoxide-ulf-entrees-{}", std::process::id()));
    std::fs::create_dir_all(&dossier).unwrap();
    let chemin = dossier.join("inv_phases.h5");
    let f = hdf5::File::create(&chemin).unwrap();
    f.new_dataset::<hdf5::types::VarLenUnicode>()
        .create("xml_header")
        .unwrap()
        .write_scalar(&xml)
        .unwrap();
    f.new_dataset_builder().with_data(&inv.kspace).create("kspace").unwrap();
    drop(f);
    let e = refus(&d.join("reg.h5"), &chemin);
    assert!(e.contains("invert_ph_polarity"), "{e}");
    std::fs::remove_dir_all(&dossier).unwrap();
}
