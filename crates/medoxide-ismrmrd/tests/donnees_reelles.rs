//! Critère 3 de docs/ULF.md (P3) : lecture de fichiers réels, comparée au code Python de référence par empreinte.
//!
//! Activé par `MEDOXIDE_ULF_DONNEES=<empreintes.json>` (écrit par `scripts/empreintes_ismrmrd.py`) ; sans cette
//! variable, le test est sauté (les données réelles ne sont pas dans le dépôt, qui est public).

use medoxide_ismrmrd::{lire, Polarite};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// SHA-256 des octets complex64 du k-space, ordre C (`[lecture, phase 1, phase 2]`, le dernier axe varie le plus vite),
/// comme `numpy.ndarray.tobytes()` : partie réelle puis imaginaire, en petit-boutiste.
fn empreinte(k: &ndarray::Array3<num_complex::Complex32>) -> String {
    let mut h = Sha256::new();
    for z in k.iter() {
        h.update(z.re.to_le_bytes());
        h.update(z.im.to_le_bytes());
    }
    h.finalize().iter().map(|o| format!("{o:02x}")).collect()
}

#[test]
fn fichiers_reels_comme_le_code_python() {
    let Ok(chemin) = std::env::var("MEDOXIDE_ULF_DONNEES") else {
        eprintln!("MEDOXIDE_ULF_DONNEES absent : test sur données réelles sauté");
        return;
    };
    let liste: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&chemin).unwrap()).unwrap();
    assert!(!liste.is_empty(), "{chemin} : aucune empreinte");
    for v in &liste {
        let fichier = v["chemin"].as_str().unwrap();
        let a = lire(std::path::Path::new(fichier)).unwrap_or_else(|e| panic!("{e}"));
        let matrice: Vec<usize> = v["matrice"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(a.kspace.shape(), matrice.as_slice(), "{fichier}");
        assert!(a.kspace.is_standard_layout());
        assert_eq!(
            empreinte(&a.kspace),
            v["kspace_sha256"].as_str().unwrap(),
            "{fichier} : k-space"
        );
        let compte: Vec<u16> = v["lectures_par_ligne"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u16)
            .collect();
        assert_eq!(
            a.lectures_par_ligne.iter().copied().collect::<Vec<_>>(),
            compte,
            "{fichier} : lignes"
        );
        assert_eq!(a.dwell_us, v["dwell_us"].as_f64().unwrap(), "{fichier} : dwell");
        assert_eq!(a.entete.porteuse_hz, v["porteuse_hz"].as_f64(), "{fichier} : porteuse");
        match v["format"].as_str().unwrap() {
            "Spinloop" => {
                assert_eq!(a.axes.as_deref(), v["axes"].as_str(), "{fichier} : axes");
                let attendue = v["read_sign"].as_str().map(|s| {
                    if s == "-1" {
                        Polarite::Inversee
                    } else {
                        Polarite::Normale
                    }
                });
                assert_eq!(a.polarite_lecture, attendue, "{fichier} : polarité");
            }
            _ => {
                assert_eq!(a.plan.as_deref(), v["plan"].as_str(), "{fichier} : plan");
                let inversee = v["invert_ro_polarity"].as_str() == Some("True");
                assert_eq!(a.lecture_retournee_a_l_ecriture, inversee, "{fichier} : polarité");
                assert_eq!(a.entete.tr_ms, v["tr_ms"].as_f64(), "{fichier} : TR");
            }
        }
        eprintln!("{fichier} : identique au code Python");
    }
}
