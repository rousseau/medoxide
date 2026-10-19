//! Table des 19 étiquettes de tissus de BOUNTI (plus le fond) et son écriture au format `dseg.tsv` de BIDS : colonnes `index`, `name`, `abbreviation`, `color`.
//!
//! Numéros, noms et couleurs : fichier `dhcp-atlas-ITKSNAP-19-labels-colour-scheme.txt` de l'atlas fœtal de KCL (licence CC0, voir `docs/BOUNTI.md`) ; dans chaque paire, le numéro de gauche précède celui de droite (impair à gauche pour les paires 1 à 12, pair à gauche pour 14 et 15 puis 16 et 17). Les abréviations sont de ce projet.

/// Une étiquette : numéro dans l'image, nom, abréviation et couleur `#RRGGBB`.
pub struct Etiquette {
    /// Valeur du voxel dans l'image d'étiquettes.
    pub index: u8,
    /// Nom complet.
    pub name: &'static str,
    /// Abréviation (`_L` : gauche, `_R` : droite).
    pub abbreviation: &'static str,
    /// Couleur d'affichage, `#RRGGBB`.
    pub color: &'static str,
}

const fn e(index: u8, name: &'static str, abbreviation: &'static str, color: &'static str) -> Etiquette {
    Etiquette {
        index,
        name,
        abbreviation,
        color,
    }
}

/// Le fond (0) et les 19 tissus, dans l'ordre des numéros.
pub const ETIQUETTES: [Etiquette; 20] = [
    e(0, "Background", "BG", "#000000"),
    e(1, "External CSF, left", "eCSF_L", "#EB4D62"),
    e(2, "External CSF, right", "eCSF_R", "#FFB700"),
    e(3, "Cortical grey matter, left", "cGM_L", "#0000FF"),
    e(4, "Cortical grey matter, right", "cGM_R", "#B100FD"),
    e(5, "Fetal white matter, left", "fWM_L", "#00FFFF"),
    e(6, "Fetal white matter, right", "fWM_R", "#B6FF4F"),
    e(7, "Lateral ventricle, left", "LV_L", "#4A8C00"),
    e(8, "Lateral ventricle, right", "LV_R", "#0033CD"),
    e(9, "Cavum septum pellucidum", "CSP", "#F45DDE"),
    e(10, "Brainstem", "BS", "#004A8C"),
    e(11, "Cerebellum, left", "Cb_L", "#66CD68"),
    e(12, "Cerebellum, right", "Cb_R", "#6868EF"),
    e(13, "Cerebellar vermis", "Vermis", "#75003C"),
    e(14, "Basal ganglia, left", "BG_L", "#EB00A1"),
    e(15, "Basal ganglia, right", "BG_R", "#FF0800"),
    e(16, "Thalamus, left", "Thal_L", "#602490"),
    e(17, "Thalamus, right", "Thal_R", "#997800"),
    e(18, "Third ventricle", "V3", "#8441FF"),
    e(19, "Fourth ventricle", "V4", "#00908B"),
];

/// Contenu du fichier `dseg.tsv` : une ligne d'en-tête puis une ligne par étiquette (fond compris), séparées par des tabulations.
pub fn dseg_tsv() -> String {
    let mut texte = String::from("index\tname\tabbreviation\tcolor\n");
    for etiquette in &ETIQUETTES {
        texte.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            etiquette.index, etiquette.name, etiquette.abbreviation, etiquette.color
        ));
    }
    texte
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{CANAUX, PARTENAIRE};

    /// Les numéros vont de 0 à 19 dans l'ordre, une seule fois chacun, avec autant de lignes que de canaux du réseau.
    #[test]
    fn the_indices_are_consecutive_and_match_the_network_channels() {
        assert_eq!(ETIQUETTES.len(), CANAUX);
        for (position, etiquette) in ETIQUETTES.iter().enumerate() {
            assert_eq!(usize::from(etiquette.index), position);
        }
    }

    /// Les paires gauche/droite de la table sont exactement celles que la fusion échange (`PARTENAIRE`) : le partenaire d'une étiquette `_L` est l'étiquette `_R` de même nom qui la suit (numéro + 1), et réciproquement ; les autres sont leur propre partenaire.
    #[test]
    fn the_left_right_pairs_are_the_ones_swapped_by_the_fusion() {
        for etiquette in &ETIQUETTES {
            let partenaire = &ETIQUETTES[PARTENAIRE[usize::from(etiquette.index)] as usize];
            if let Some(base) = etiquette.abbreviation.strip_suffix("_L") {
                assert_eq!(partenaire.abbreviation, format!("{base}_R"), "{}", etiquette.name);
                assert_eq!(partenaire.index, etiquette.index + 1);
            } else if let Some(base) = etiquette.abbreviation.strip_suffix("_R") {
                assert_eq!(partenaire.abbreviation, format!("{base}_L"), "{}", etiquette.name);
                assert_eq!(partenaire.index, etiquette.index - 1);
            } else {
                assert_eq!(partenaire.index, etiquette.index, "{} est médiane", etiquette.name);
            }
        }
    }

    /// Les couleurs sont au format `#RRGGBB` (majuscules), distinctes, et les noms et abréviations ne se répètent pas.
    #[test]
    fn colors_names_and_abbreviations_are_well_formed_and_unique() {
        let mut couleurs = std::collections::HashSet::new();
        let mut noms = std::collections::HashSet::new();
        let mut abreviations = std::collections::HashSet::new();
        for etiquette in &ETIQUETTES {
            let c = etiquette.color;
            assert_eq!(c.len(), 7, "{c}");
            assert!(
                c.starts_with('#') && c[1..].chars().all(|h| h.is_ascii_digit() || ('A'..='F').contains(&h)),
                "{c}"
            );
            assert!(couleurs.insert(c), "couleur répétée {c}");
            assert!(noms.insert(etiquette.name), "nom répété {}", etiquette.name);
            assert!(
                abreviations.insert(etiquette.abbreviation),
                "abréviation répétée {}",
                etiquette.abbreviation
            );
        }
    }

    /// Le fichier `dseg.tsv` : en-tête `index name abbreviation color`, vingt lignes de quatre champs séparés par des tabulations, terminées par un saut de ligne, et la ligne de l'étiquette 17 contient le thalamus droit de la table du dépôt de l'atlas.
    #[test]
    fn the_tsv_has_a_header_and_one_row_per_label() {
        let texte = dseg_tsv();
        assert!(texte.ends_with('\n'));
        let lignes: Vec<&str> = texte.lines().collect();
        assert_eq!(lignes.len(), 21);
        assert_eq!(lignes[0], "index\tname\tabbreviation\tcolor");
        for ligne in &lignes[1..] {
            assert_eq!(ligne.split('\t').count(), 4, "{ligne}");
        }
        assert_eq!(lignes[18], "17\tThalamus, right\tThal_R\t#997800");
        assert_eq!(lignes[1], "0\tBackground\tBG\t#000000");
    }

    /// Les 20 couleurs de la table sont celles du fichier de l'atlas (RGB en décimal, recopiées du fichier brut).
    #[test]
    fn the_colors_come_from_the_atlas_file() {
        #[rustfmt::skip]
        let atlas: [[u8; 3]; 20] = [
            [0, 0, 0], [235, 77, 98], [255, 183, 0], [0, 0, 255], [177, 0, 253], [0, 255, 255], [182, 255, 79], [74, 140, 0],
            [0, 51, 205], [244, 93, 222], [0, 74, 140], [102, 205, 104], [104, 104, 239], [117, 0, 60], [235, 0, 161], [255, 8, 0],
            [96, 36, 144], [153, 120, 0], [132, 65, 255], [0, 144, 139],
        ];
        for (index, rgb) in atlas.iter().enumerate() {
            let couleur = ETIQUETTES[index].color;
            let lu: Vec<u8> = (0..3)
                .map(|k| u8::from_str_radix(&couleur[1 + 2 * k..3 + 2 * k], 16).unwrap())
                .collect();
            assert_eq!(&lu, rgb, "étiquette {index}");
        }
    }

    /// Chaque nom désigne la bonne structure (mot-clé de la table de l'atlas) et son côté : les étiquettes `_L` finissent par « , left », les `_R` par « , right ».
    #[test]
    fn the_names_designate_the_right_structure_and_side() {
        let mots = [
            "Background",
            "External CSF",
            "External CSF",
            "Cortical",
            "Cortical",
            "white matter",
            "white matter",
            "Lateral ventricle",
            "Lateral ventricle",
            "Cavum",
            "Brainstem",
            "Cerebellum",
            "Cerebellum",
            "vermis",
            "Basal ganglia",
            "Basal ganglia",
            "Thalamus",
            "Thalamus",
            "Third",
            "Fourth",
        ];
        for (etiquette, mot) in ETIQUETTES.iter().zip(mots) {
            assert!(etiquette.name.contains(mot), "{} ne contient pas {mot}", etiquette.name);
            if etiquette.abbreviation.ends_with("_L") {
                assert!(etiquette.name.ends_with(", left"), "{}", etiquette.name);
            }
            if etiquette.abbreviation.ends_with("_R") {
                assert!(etiquette.name.ends_with(", right"), "{}", etiquette.name);
            }
        }
    }
}
