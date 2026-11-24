//! Critères 1 et 2 de docs/ULF.md (P3) sur de petits fichiers synthétiques, écrits par les écrivains Python d'origine
//! (`scripts/ecrire_fixtures_ismrmrd.py`) avec les valeurs que le code Python de référence en relit.

use std::path::{Path, PathBuf};

use medoxide_ismrmrd::{lire, Acquisition, Format, Polarite};
use serde_json::Value;

fn donnees(nom: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees").join(nom)
}

fn attendu(format: &str) -> Value {
    let texte = std::fs::read_to_string(donnees(&format!("{format}_attendu.json"))).unwrap();
    serde_json::from_str(&texte).unwrap()
}

fn nombres(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// Compare le k-space (ordre C, `[lecture, phase 1, phase 2]`) et le compte des lignes ; rend l'écart relatif maximal.
fn ecart_kspace(a: &Acquisition, v: &Value) -> f64 {
    let forme: Vec<usize> = v["forme"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(a.kspace.shape(), forme.as_slice());
    let (re, im) = (nombres(&v["kspace_re"]), nombres(&v["kspace_im"]));
    let mut ecart = 0f64;
    for (i, z) in a.kspace.iter().enumerate() {
        let attendu = (re[i] * re[i] + im[i] * im[i]).sqrt().max(1e-30);
        let d = ((z.re as f64 - re[i]).powi(2) + (z.im as f64 - im[i]).powi(2)).sqrt();
        ecart = ecart.max(d / attendu);
    }
    let compte: Vec<u16> = v["lectures_par_ligne"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u16)
        .collect();
    assert_eq!(a.lectures_par_ligne.iter().copied().collect::<Vec<_>>(), compte);
    ecart
}

fn entete_commun(a: &Acquisition, v: &Value) {
    let matrice: Vec<usize> = v["matrice"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(a.entete.matrice.to_vec(), matrice);
    assert_eq!(a.entete.champ_de_vue_mm.to_vec(), nombres(&v["champ_de_vue_mm"]));
    let centre: Vec<usize> = v["centre"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(a.entete.centre.map(|c| c.to_vec()), Some(centre));
    assert_eq!(a.entete.porteuse_hz, v["porteuse_hz"].as_f64());
    assert_eq!(a.dwell_us, v["dwell_us"].as_f64().unwrap());
    assert_eq!(a.axes.as_deref(), v["axes"].as_str());
    assert_eq!(a.plan.as_deref(), v["plan"].as_str());
    assert_eq!(
        a.lecture_retournee_a_l_ecriture,
        v["lecture_retournee_a_l_ecriture"].as_bool().unwrap()
    );
    let polarite = match v["polarite_lecture"].as_str().unwrap() {
        "Inversee" => Polarite::Inversee,
        _ => Polarite::Normale,
    };
    assert_eq!(a.polarite_lecture, Some(polarite));
}

/// Critère 1 : k-space égal à `spinloop_recon.cartesian.grid` à 1e-6 relatif (moyennes f32 / f64), lignes répétées
/// moyennées, train d'équilibre ignoré, ligne non acquise à 0 ; en-tête égal à `spinloop_recon.mrd.read`.
#[test]
fn spinloop_comme_la_reference_python() {
    let a = lire(&donnees("spinloop.h5")).unwrap();
    let v = attendu("spinloop");
    assert_eq!(a.format, Format::Spinloop);
    entete_commun(&a, &v);
    let ecart = ecart_kspace(&a, &v);
    assert!(ecart < 1e-6, "écart relatif maximal {ecart:e}");
    // précondition du test : le fichier contient bien une ligne moyennée et une ligne non acquise
    assert!(a.lectures_par_ligne.iter().any(|&c| c == 2) && a.lectures_par_ligne.iter().any(|&c| c == 0));
}

/// Critère 2 : k-space égal au bit près à h5py ; lignes acquises = `np.any(k != 0, axis=0)` ; en-tête de fmi.
#[test]
fn fmi_comme_h5py() {
    let a = lire(&donnees("fmi.h5")).unwrap();
    let v = attendu("fmi");
    assert_eq!(a.format, Format::Fmi);
    entete_commun(&a, &v);
    assert_eq!(ecart_kspace(&a, &v), 0.0);
    assert_eq!(a.entete.tr_ms, v["tr_ms"].as_f64());
    assert_eq!(a.entete.parametre_nombre("echoShift"), Some(0.2));
}

#[test]
fn refuse_un_fichier_d_un_autre_format() {
    let dossier = std::env::temp_dir().join(format!("medoxide-ismrmrd-{}", std::process::id()));
    std::fs::create_dir_all(&dossier).unwrap();
    let chemin = dossier.join("autre.h5");
    let f = hdf5::File::create(&chemin).unwrap();
    f.new_dataset::<f32>().shape([2]).create("autre").unwrap();
    drop(f);
    let erreur = lire(&chemin).unwrap_err();
    assert!(erreur.to_string().contains("ni dataset/data"), "{erreur}");
    std::fs::remove_dir_all(&dossier).unwrap();
}

/// Copie de `fmi.h5` dont la ligne `invert_ro_polarity` de l'en-tête est remplacée par `parametres` (lignes XML).
fn fmi_modifie(nom: &str, parametres: &str) -> (PathBuf, Acquisition) {
    let source = lire(&donnees("fmi.h5")).unwrap();
    let ligne = "  <userParameterString><name>invert_ro_polarity</name><value>True</value></userParameterString>\n";
    assert!(source.xml.contains(ligne));
    let xml: hdf5::types::VarLenUnicode = source.xml.replace(ligne, parametres).parse().unwrap();
    let dossier = std::env::temp_dir().join(format!("medoxide-ismrmrd-{}-{nom}", std::process::id()));
    std::fs::create_dir_all(&dossier).unwrap();
    let chemin = dossier.join("fmi.h5");
    let f = hdf5::File::create(&chemin).unwrap();
    f.new_dataset::<hdf5::types::VarLenUnicode>()
        .create("xml_header")
        .unwrap()
        .write_scalar(&xml)
        .unwrap();
    f.new_dataset_builder()
        .with_data(&source.kspace)
        .create("kspace")
        .unwrap();
    drop(f);
    (dossier, source)
}

/// Paramètres absents : faux, comme l'écrivain de fmi (`inv_arr_ro.size > 0 else False`).
#[test]
fn fmi_polarites_absentes_valent_faux() {
    let (dossier, source) = fmi_modifie("absent", "");
    let a = lire(&dossier.join("fmi.h5")).unwrap();
    assert_eq!(a.polarite_lecture, Some(Polarite::Normale));
    assert!(!a.lecture_retournee_a_l_ecriture);
    assert!(!a.phases_retournees_a_l_ecriture);
    assert_eq!(a.kspace, source.kspace);
    std::fs::remove_dir_all(&dossier).unwrap();
}

/// `invert_ph_polarity` = True : l'écrivain a retourné les axes de phase, signalé ; valeur inconnue refusée.
#[test]
fn fmi_phases_retournees_signalees() {
    let ligne =
        |n: &str, v: &str| format!("  <userParameterString><name>{n}</name><value>{v}</value></userParameterString>\n");
    let (dossier, _) = fmi_modifie(
        "phases",
        &(ligne("invert_ro_polarity", "False") + &ligne("invert_ph_polarity", "True")),
    );
    let a = lire(&dossier.join("fmi.h5")).unwrap();
    assert_eq!(a.polarite_lecture, Some(Polarite::Normale));
    assert!(a.phases_retournees_a_l_ecriture);
    std::fs::remove_dir_all(&dossier).unwrap();

    let (dossier, _) = fmi_modifie("inconnu", &ligne("invert_ph_polarity", "peut-être"));
    let e = lire(&dossier.join("fmi.h5")).unwrap_err();
    assert!(e.to_string().contains("invert_ph_polarity"), "{e}");
    std::fs::remove_dir_all(&dossier).unwrap();
}
