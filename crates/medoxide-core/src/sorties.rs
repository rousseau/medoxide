//! Lecture des tableaux que `medx` écrit (contrat de données de guioxide, `docs/donnees.md`) : qualité par coupe (`*_desc-svr_slicequality.tsv`), mouvement par coupe (`*_desc-svr_motion.tsv`) et table des étiquettes (`*_dseg.tsv`),
//! et le mouvement par coupe qu'on en déduit ([`mouvement_par_coupe`]).
//!
//! Format commun : TSV avec ligne d'en-tête, séparateur tabulation, décimal `.`, `n/a` (ou vide) pour une valeur absente. Les colonnes sont repérées par leur nom, dans n'importe quel ordre ; une colonne en trop est ignorée.
//! Sans Burn et sans dépendance : un lecteur (guioxide) n'a pas à tirer un module de calcul.

use std::path::{Path, PathBuf};

/// Erreurs de lecture d'un tableau.
#[derive(Debug)]
pub enum ErreurTableau {
    /// Le fichier ne se lit pas (absent, droits, texte invalide).
    Lecture {
        /// Fichier concerné.
        chemin: PathBuf,
        /// Message de l'erreur.
        message: String,
    },
    /// Le contenu ne suit pas le format : colonne obligatoire absente, valeur invalide, ligne trop courte.
    Format {
        /// Fichier concerné.
        chemin: PathBuf,
        /// Ce qui ne va pas, avec le numéro de ligne du fichier (l'en-tête est la ligne 1).
        message: String,
    },
}

impl std::fmt::Display for ErreurTableau {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurTableau::Lecture { chemin, message } | ErreurTableau::Format { chemin, message } => {
                write!(f, "{} : {message}", chemin.display())
            }
        }
    }
}

impl std::error::Error for ErreurTableau {}

/// Un tableau TSV lu en entier : les fichiers lus ici sont petits (quelques milliers de lignes).
struct Tableau<'a> {
    chemin: &'a Path,
    colonnes: Vec<String>,
    /// Lignes (hors en-tête et lignes vides) avec leur numéro dans le fichier, à partir de 1.
    lignes: Vec<(usize, Vec<String>)>,
}

impl<'a> Tableau<'a> {
    fn lire(chemin: &'a Path) -> Result<Tableau<'a>, ErreurTableau> {
        let texte = std::fs::read_to_string(chemin).map_err(|e| ErreurTableau::Lecture {
            chemin: chemin.to_path_buf(),
            message: e.to_string(),
        })?;
        let decouper = |l: &str| l.split('\t').map(|c| c.trim().to_string()).collect::<Vec<_>>();
        let mut lignes = texte.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
        let colonnes = lignes.next().map(|(_, l)| decouper(l)).unwrap_or_default();
        Ok(Tableau {
            chemin,
            colonnes,
            lignes: lignes.map(|(i, l)| (i + 1, decouper(l))).collect(),
        })
    }

    fn erreur(&self, ligne: usize, message: impl std::fmt::Display) -> ErreurTableau {
        ErreurTableau::Format {
            chemin: self.chemin.to_path_buf(),
            message: format!("ligne {ligne} : {message}"),
        }
    }

    /// Position de la colonne `nom`, ou une erreur si elle manque.
    fn colonne(&self, nom: &str) -> Result<usize, ErreurTableau> {
        self.colonnes
            .iter()
            .position(|c| c == nom)
            .ok_or_else(|| ErreurTableau::Format {
                chemin: self.chemin.to_path_buf(),
                message: format!("colonne « {nom} » absente"),
            })
    }

    /// Position de la colonne `nom` si elle existe.
    fn colonne_facultative(&self, nom: &str) -> Option<usize> {
        self.colonnes.iter().position(|c| c == nom)
    }

    /// Valeur de la colonne `colonne` d'une ligne ; `None` si elle est vide ou vaut `n/a`.
    fn valeur(ligne: &[String], colonne: usize) -> Option<&str> {
        let v = ligne.get(colonne)?.as_str();
        (!v.is_empty() && v != "n/a").then_some(v)
    }

    /// Valeur obligatoire (ni vide ni `n/a`) d'une colonne.
    fn obligatoire<'l>(&self, ligne: &'l [String], numero: usize, colonne: usize) -> Result<&'l str, ErreurTableau> {
        Self::valeur(ligne, colonne).ok_or_else(|| {
            self.erreur(
                numero,
                format!("colonne « {} » vide ou absente", self.colonnes[colonne]),
            )
        })
    }

    /// Nombre lu dans une valeur.
    fn nombre<T: std::str::FromStr>(&self, texte: &str, colonne: usize, numero: usize) -> Result<T, ErreurTableau> {
        texte
            .parse()
            .map_err(|_| self.erreur(numero, format!("« {texte} » invalide pour {}", self.colonnes[colonne])))
    }
}

/// Une ligne de `*_slicequality.tsv` : la qualité d'une coupe (voir `ReconstructionOutput::write_quality` dans `medoxide-svr`).
#[derive(Debug, Clone, PartialEq)]
pub struct QualiteCoupe {
    /// Nom du fichier de la pile, sans extension.
    pub stack: String,
    /// Indice de la coupe dans la pile (0 = première).
    pub slice: usize,
    /// Coupe recalée.
    pub registered: bool,
    /// Concordance avant le recalage ; `None` (`n/a`) pour une coupe non recalée.
    pub score_before: Option<f64>,
    /// Concordance après le recalage.
    pub score_after: Option<f64>,
    /// Nombre de recalages.
    pub n_registrations: Option<u32>,
    /// Amplitude de la correction de mouvement (mm).
    pub correction_rms_mm: Option<f64>,
    /// Coupe retirée de la reconstruction.
    pub excluded: bool,
    /// Motif de l'exclusion (`faible_concordance`, `hors_masque`…) ; `None` (`n/a`) si la coupe n'est pas exclue.
    pub reason: Option<String>,
}

/// Lit `*_slicequality.tsv`. Colonnes obligatoires : `stack` et `slice` ; les autres peuvent manquer (valeur absente ; coupe ni recalée ni exclue).
///
/// # Erreurs
/// [`ErreurTableau::Lecture`] si le fichier ne se lit pas ; [`ErreurTableau::Format`] (avec le numéro de ligne) si une colonne obligatoire manque ou si une valeur n'est pas un nombre.
pub fn lire_qualite(chemin: &Path) -> Result<Vec<QualiteCoupe>, ErreurTableau> {
    let t = Tableau::lire(chemin)?;
    let (c_stack, c_slice) = (t.colonne("stack")?, t.colonne("slice")?);
    let c = |nom: &str| t.colonne_facultative(nom);
    let (c_registered, c_avant, c_apres) = (c("registered"), c("score_before"), c("score_after"));
    let (c_n, c_rms, c_excluded, c_reason) = (c("n_registrations"), c("correction_rms_mm"), c("excluded"), c("reason"));
    let mut sortie = Vec::with_capacity(t.lignes.len());
    for (numero, ligne) in &t.lignes {
        let numero = *numero;
        let reel = |colonne: Option<usize>| -> Result<Option<f64>, ErreurTableau> {
            colonne
                .and_then(|i| Tableau::valeur(ligne, i).map(|v| (i, v)))
                .map(|(i, v)| t.nombre(v, i, numero))
                .transpose()
        };
        let drapeau = |colonne: Option<usize>| colonne.and_then(|i| Tableau::valeur(ligne, i)) == Some("1");
        sortie.push(QualiteCoupe {
            stack: t.obligatoire(ligne, numero, c_stack)?.to_string(),
            slice: t.nombre(t.obligatoire(ligne, numero, c_slice)?, c_slice, numero)?,
            registered: drapeau(c_registered),
            score_before: reel(c_avant)?,
            score_after: reel(c_apres)?,
            n_registrations: c_n
                .and_then(|i| Tableau::valeur(ligne, i).map(|v| (i, v)))
                .map(|(i, v)| t.nombre(v, i, numero))
                .transpose()?,
            correction_rms_mm: reel(c_rms)?,
            excluded: drapeau(c_excluded),
            reason: c_reason.and_then(|i| Tableau::valeur(ligne, i)).map(str::to_string),
        });
    }
    Ok(sortie)
}

/// Une ligne de `*_motion.tsv` : le mouvement rigide d'une coupe.
///
/// `matrice` est `M` (4 × 4, `matrice[ligne][colonne]`, mm) : le mouvement de la coupe, **absolu** par rapport à la géométrie d'en-tête de sa pile et composé dans le repère monde (RAS+). La coupe `k` d'une pile d'affine d'en-tête `A` est placée par `M · A · T(0, 0, k)`.
#[derive(Debug, Clone, PartialEq)]
pub struct MouvementCoupe {
    /// Nom du fichier de la pile, sans extension.
    pub stack: String,
    /// Indice de la coupe dans la pile (0 = première).
    pub slice: usize,
    /// Mouvement rigide `M`.
    pub matrice: [[f64; 4]; 4],
}

/// Lit `*_motion.tsv` : `stack`, `slice`, puis la matrice en colonnes `m<ligne><colonne>` (le fichier les range colonne par colonne : `m00 m10 m20 m30 m01 …`, mais l'ordre est sans importance ici, les colonnes sont repérées par leur nom).
///
/// # Erreurs
/// [`ErreurTableau::Lecture`] ; [`ErreurTableau::Format`] (avec le numéro de ligne) si une colonne (`stack`, `slice`, `m00` … `m33`) manque, ou si une valeur n'est pas un nombre.
pub fn lire_mouvement(chemin: &Path) -> Result<Vec<MouvementCoupe>, ErreurTableau> {
    let t = Tableau::lire(chemin)?;
    let (c_stack, c_slice) = (t.colonne("stack")?, t.colonne("slice")?);
    let mut positions = [[0usize; 4]; 4];
    for (l, ligne) in positions.iter_mut().enumerate() {
        for (c, p) in ligne.iter_mut().enumerate() {
            *p = t.colonne(&format!("m{l}{c}"))?;
        }
    }
    let mut sortie = Vec::with_capacity(t.lignes.len());
    for (numero, ligne) in &t.lignes {
        let mut matrice = [[0.0; 4]; 4];
        for (l, valeurs) in matrice.iter_mut().enumerate() {
            for (c, x) in valeurs.iter_mut().enumerate() {
                *x = t.nombre(
                    t.obligatoire(ligne, *numero, positions[l][c])?,
                    positions[l][c],
                    *numero,
                )?;
            }
        }
        sortie.push(MouvementCoupe {
            stack: t.obligatoire(ligne, *numero, c_stack)?.to_string(),
            slice: t.nombre(t.obligatoire(ligne, *numero, c_slice)?, c_slice, *numero)?,
            matrice,
        });
    }
    Ok(sortie)
}

/// Mouvement d'une coupe, résumé pour une courbe de mouvement.
#[derive(Debug, Clone, PartialEq)]
pub struct PointMouvement {
    /// Nom du fichier de la pile, sans extension.
    pub stack: String,
    /// Indice de la coupe dans la pile.
    pub slice: usize,
    /// Déplacement du centre de la coupe, `‖M·c − c‖` (mm).
    pub translation_mm: f64,
    /// Angle de la rotation de `M` (degrés, de 0 à 180).
    pub rotation_deg: f64,
    /// Coupe retirée de la reconstruction (lue dans la qualité par coupe).
    pub excluded: bool,
}

/// Mouvement par coupe : déplacement du centre de chaque coupe et angle de rotation de `M`.
///
/// Le centre `c` de la coupe `k` est, dans la géométrie d'en-tête de sa pile, le point du plan de la coupe au milieu des voxels : `A · ((nx − 1)/2, (ny − 1)/2, k, 1)`. Le déplacement est `‖M·c − c‖` (mm, `M` appliquée en coordonnées homogènes) ; l'angle est celui de la rotation de `M`, `arccos((trace(R) − 1)/2)`, en degrés.
///
/// `geometrie` donne, pour le nom d'une pile, ses dimensions et son affine d'en-tête (voir [`crate::affine_nifti`] et [`crate::volume_info`]) ; les coupes d'une pile dont elle ne donne rien sont omises. `qualite` fournit les coupes exclues de la reconstruction (`excluded` vrai) ; sans elle, aucune coupe n'est marquée exclue.
pub fn mouvement_par_coupe(
    mouvement: &[MouvementCoupe],
    geometrie: impl Fn(&str) -> Option<([usize; 3], [[f64; 4]; 4])>,
    qualite: Option<&[QualiteCoupe]>,
) -> Vec<PointMouvement> {
    let exclues: std::collections::HashSet<(&str, usize)> = qualite
        .unwrap_or_default()
        .iter()
        .filter(|q| q.excluded)
        .map(|q| (q.stack.as_str(), q.slice))
        .collect();
    let appliquer = |m: &[[f64; 4]; 4], p: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|l| m[l][0] * p[0] + m[l][1] * p[1] + m[l][2] * p[2] + m[l][3])
    };
    mouvement
        .iter()
        .filter_map(|ligne| {
            let (dim, affine) = geometrie(&ligne.stack)?;
            let centre = appliquer(
                &affine,
                [
                    (dim[0] as f64 - 1.0) / 2.0,
                    (dim[1] as f64 - 1.0) / 2.0,
                    ligne.slice as f64,
                ],
            );
            let deplace = appliquer(&ligne.matrice, centre);
            let translation_mm = (0..3).map(|l| (deplace[l] - centre[l]).powi(2)).sum::<f64>().sqrt();
            let trace = ligne.matrice[0][0] + ligne.matrice[1][1] + ligne.matrice[2][2];
            Some(PointMouvement {
                stack: ligne.stack.clone(),
                slice: ligne.slice,
                translation_mm,
                rotation_deg: ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees(),
                excluded: exclues.contains(&(ligne.stack.as_str(), ligne.slice)),
            })
        })
        .collect()
}

/// Une ligne de `*_dseg.tsv` : une étiquette d'une segmentation (format `dseg.tsv` de BIDS).
#[derive(Debug, Clone, PartialEq)]
pub struct EtiquetteDseg {
    /// Valeur du voxel dans l'image d'étiquettes (0 : le fond).
    pub index: u32,
    /// Nom complet ; vide s'il est absent.
    pub name: String,
    /// Abréviation.
    pub abbreviation: Option<String>,
    /// Couleur RVB, lue dans `#RRGGBB` ; `None` si elle est absente ou n'a pas cette forme (c'est une aide à l'affichage : le choix d'une couleur par défaut revient à l'appelant).
    pub color: Option<[u8; 3]>,
}

/// `#rrggbb` → RVB.
fn lire_couleur(texte: &str) -> Option<[u8; 3]> {
    let h = texte.strip_prefix('#')?;
    if h.len() != 6 || !h.is_ascii() {
        return None;
    }
    let canal = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok();
    Some([canal(0)?, canal(2)?, canal(4)?])
}

/// Lit `*_dseg.tsv` : `index` (obligatoire), `name`, `abbreviation`, `color`. Une ligne sans `index` est ignorée.
///
/// # Erreurs
/// [`ErreurTableau::Lecture`] ; [`ErreurTableau::Format`] si la colonne `index` manque ou si un indice n'est pas un entier positif.
pub fn lire_dseg_tsv(chemin: &Path) -> Result<Vec<EtiquetteDseg>, ErreurTableau> {
    let t = Tableau::lire(chemin)?;
    let c_index = t.colonne("index")?;
    let (c_nom, c_abreviation, c_couleur) = (
        t.colonne_facultative("name"),
        t.colonne_facultative("abbreviation"),
        t.colonne_facultative("color"),
    );
    let mut sortie = Vec::with_capacity(t.lignes.len());
    for (numero, ligne) in &t.lignes {
        let Some(index) = Tableau::valeur(ligne, c_index) else {
            continue;
        };
        let texte = |colonne: Option<usize>| colonne.and_then(|i| Tableau::valeur(ligne, i));
        sortie.push(EtiquetteDseg {
            index: t.nombre(index, c_index, *numero)?,
            name: texte(c_nom).unwrap_or_default().to_string(),
            abbreviation: texte(c_abreviation).map(str::to_string),
            color: texte(c_couleur).and_then(lire_couleur),
        });
    }
    Ok(sortie)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dossier temporaire propre à un test (nom + processus), supprimé en fin de test.
    struct Temp(PathBuf);
    impl Temp {
        fn new(nom: &str) -> Temp {
            let d = std::env::temp_dir().join(format!("medoxide_sorties_{nom}_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Temp(d)
        }
        fn ecrire(&self, nom: &str, texte: &str) -> PathBuf {
            let chemin = self.0.join(nom);
            std::fs::write(&chemin, texte).unwrap();
            chemin
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const ENTETE_MOUVEMENT: &str =
        "stack\tslice\tm00\tm10\tm20\tm30\tm01\tm11\tm21\tm31\tm02\tm12\tm22\tm32\tm03\tm13\tm23\tm33";

    /// Ligne de mouvement : `stack`, `slice`, puis la matrice `m[ligne][colonne]` rangée colonne par colonne.
    fn ligne_mouvement(stack: &str, slice: usize, m: [[f64; 4]; 4]) -> String {
        let valeurs: Vec<String> = (0..4)
            .flat_map(|c| (0..4).map(move |l| m[l][c]))
            .map(|v| format!("{v:?}"))
            .collect();
        format!("{stack}\t{slice}\t{}", valeurs.join("\t"))
    }

    /// Qualité selon le contrat : en-tête de `write_quality`, `n/a` pour les valeurs absentes, motif d'exclusion.
    #[test]
    fn the_quality_table_follows_the_contract() {
        let t = Temp::new("qualite");
        let c = t.ecrire(
            "q.tsv",
            "stack\tslice\tregistered\tscore_before\tscore_after\tn_registrations\tcorrection_rms_mm\texcluded\treason\n\
             p1\t0\t1\t0.5\t0.8\t2\t1.25\t0\tn/a\n\
             p1\t1\t0\tn/a\tn/a\t0\t0.0\t1\thors_masque\n",
        );
        let q = lire_qualite(&c).unwrap();
        assert_eq!(
            q[0],
            QualiteCoupe {
                stack: "p1".into(),
                slice: 0,
                registered: true,
                score_before: Some(0.5),
                score_after: Some(0.8),
                n_registrations: Some(2),
                correction_rms_mm: Some(1.25),
                excluded: false,
                reason: None,
            }
        );
        assert_eq!(
            (
                q[1].registered,
                q[1].score_before,
                q[1].score_after,
                q[1].n_registrations
            ),
            (false, None, None, Some(0))
        );
        assert_eq!((q[1].excluded, q[1].reason.as_deref()), (true, Some("hors_masque")));
    }

    /// Colonnes facultatives absentes (valeur absente), colonnes dans un autre ordre, ligne vide et fin de ligne Windows ; une valeur invalide donne le fichier et le numéro de ligne.
    #[test]
    fn the_quality_reader_is_tolerant_and_names_the_line_of_an_error() {
        let t = Temp::new("qualite_tolerance");
        let c = t.ecrire("q.tsv", "slice\tstack\tscore_after\r\n0\tp1\t0.5\r\n\r\n1\tp1\t\r\n");
        let q = lire_qualite(&c).unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!((q[0].slice, q[0].score_after, q[0].registered), (0, Some(0.5), false));
        assert_eq!(q[1].score_after, None);
        let c = t.ecrire("mauvais.tsv", "stack\tslice\tscore_after\np1\t0\t0.5\np1\tdeux\t0.5\n");
        let e = lire_qualite(&c).unwrap_err().to_string();
        assert!(
            e.contains("mauvais.tsv") && e.contains("ligne 3") && e.contains("deux"),
            "{e}"
        );
        let c = t.ecrire("sans_stack.tsv", "slice\n0\n");
        let e = lire_qualite(&c).unwrap_err().to_string();
        assert!(e.contains("colonne « stack » absente"), "{e}");
        let e = lire_qualite(&t.0.join("absent.tsv")).unwrap_err();
        assert!(matches!(e, ErreurTableau::Lecture { .. }));
    }

    /// Le mouvement relu est celui qui a été écrit, en ordre colonne : une matrice non symétrique (translation et rotation) garde ses lignes et ses colonnes.
    #[test]
    fn the_motion_table_is_read_in_column_order() {
        let t = Temp::new("mouvement");
        let m = [
            [0.0, -1.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let c = t.ecrire(
            "m.tsv",
            &format!("{ENTETE_MOUVEMENT}\n{}\n", ligne_mouvement("sub-X_T2w", 3, m)),
        );
        let lu = lire_mouvement(&c).unwrap();
        assert_eq!(
            lu,
            vec![MouvementCoupe {
                stack: "sub-X_T2w".into(),
                slice: 3,
                matrice: m
            }]
        );
        let c = t.ecrire("incomplet.tsv", "stack\tslice\tm00\np\t0\t1\n");
        assert!(lire_mouvement(&c)
            .unwrap_err()
            .to_string()
            .contains("colonne « m01 » absente"));
    }

    /// Mouvement par coupe : une translation pure de (3, 4, 0) mm donne 5 mm et 0° ; une rotation de 90° autour de z passant par l'origine déplace le centre `c = (10, 0, k)` de ‖(0, 10, 0) − (10, 0, 0)‖ = 10√2 ; les coupes exclues sont lues dans la qualité ; une pile sans géométrie est omise.
    #[test]
    fn the_per_slice_motion_is_the_displacement_of_the_slice_centre_and_the_rotation_angle() {
        // pile 21 × 3 × 4 voxels, affine = identité : centre de la coupe k = (10, 1, k)
        let geometrie = |nom: &str| {
            (nom == "a").then_some((
                [21usize, 3, 4],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ))
        };
        let identite = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut translation = identite;
        (translation[0][3], translation[1][3]) = (3.0, 4.0);
        let rotation = [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mouvement = vec![
            MouvementCoupe {
                stack: "a".into(),
                slice: 0,
                matrice: translation,
            },
            MouvementCoupe {
                stack: "a".into(),
                slice: 2,
                matrice: rotation,
            },
            MouvementCoupe {
                stack: "inconnue".into(),
                slice: 0,
                matrice: identite,
            },
        ];
        let qualite = vec![QualiteCoupe {
            stack: "a".into(),
            slice: 2,
            registered: true,
            score_before: None,
            score_after: None,
            n_registrations: None,
            correction_rms_mm: None,
            excluded: true,
            reason: Some("faible_concordance".into()),
        }];
        let points = mouvement_par_coupe(&mouvement, geometrie, Some(&qualite));
        assert_eq!(points.len(), 2, "la pile sans géométrie est omise");
        assert!((points[0].translation_mm - 5.0).abs() < 1e-12 && points[0].rotation_deg == 0.0);
        assert!(!points[0].excluded);
        // centre (10, 1, 2) → (−1, 10, 2) : déplacement ‖(−11, 9, 0)‖
        assert!((points[1].translation_mm - (121.0_f64 + 81.0).sqrt()).abs() < 1e-12);
        assert!((points[1].rotation_deg - 90.0).abs() < 1e-9);
        assert!(points[1].excluded);
        let sans_qualite = mouvement_par_coupe(&mouvement, geometrie, None);
        assert!(sans_qualite.iter().all(|p| !p.excluded));
    }

    /// Le centre est pris dans la géométrie d'en-tête de la pile : avec un affine qui échelonne et décale, une translation du centre de la coupe se mesure en mm du monde, et un mouvement qui ne bouge pas le centre (rotation autour de lui) donne 0 mm avec son angle.
    #[test]
    fn the_slice_centre_comes_from_the_header_geometry() {
        // voxels de (2, 2, 5) mm, origine (100, 0, −10) : centre de la coupe 1 d'une pile 5 × 3 × 2 = (104, 2, −5)
        let affine = [
            [2.0, 0.0, 0.0, 100.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let geometrie = |_: &str| Some(([5usize, 3, 2], affine));
        // rotation de 90° autour de l'axe z passant par le centre (104, 2) : M = T(c) · R · T(−c)
        let m = [
            [0.0, -1.0, 0.0, 104.0 + 2.0],
            [1.0, 0.0, 0.0, 2.0 - 104.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let pas_de_deplacement = vec![MouvementCoupe {
            stack: "p".into(),
            slice: 1,
            matrice: m,
        }];
        let p = &mouvement_par_coupe(&pas_de_deplacement, geometrie, None)[0];
        assert!(p.translation_mm < 1e-9, "{}", p.translation_mm);
        assert!((p.rotation_deg - 90.0).abs() < 1e-9);
        // une translation de 3 mm selon z ne dépend pas de la coupe ni de l'échelle de la pile
        let mut t = [[0.0; 4]; 4];
        for (i, ligne) in t.iter_mut().enumerate() {
            ligne[i] = 1.0;
        }
        t[2][3] = 3.0;
        let bouge = vec![MouvementCoupe {
            stack: "p".into(),
            slice: 0,
            matrice: t,
        }];
        assert!((mouvement_par_coupe(&bouge, geometrie, None)[0].translation_mm - 3.0).abs() < 1e-12);
    }

    /// Table des étiquettes : les 4 colonnes de `dseg.tsv` ; une couleur absente ou invalide donne `None` ; une colonne `index` absente est une erreur.
    #[test]
    fn the_dseg_table_is_read_with_its_colours() {
        let t = Temp::new("dseg");
        let c = t.ecrire(
            "d.tsv",
            "index\tname\tabbreviation\tcolor\n0\tBackground\tBG\t#000000\n17\tThalamus, right\tThal_R\t#997800\n18\tSans couleur\tn/a\tn/a\n19\tCouleur fausse\tX\tbleu\n",
        );
        let e = lire_dseg_tsv(&c).unwrap();
        assert_eq!(e.len(), 4);
        assert_eq!(
            e[1],
            EtiquetteDseg {
                index: 17,
                name: "Thalamus, right".into(),
                abbreviation: Some("Thal_R".into()),
                color: Some([0x99, 0x78, 0x00]),
            }
        );
        assert_eq!((e[2].abbreviation.clone(), e[2].color), (None, None));
        assert_eq!(e[3].color, None);
        let c = t.ecrire("sans_index.tsv", "name\nX\n");
        assert!(lire_dseg_tsv(&c)
            .unwrap_err()
            .to_string()
            .contains("colonne « index » absente"));
        let c = t.ecrire("mauvais_index.tsv", "index\tname\nx\tX\n");
        assert!(lire_dseg_tsv(&c).unwrap_err().to_string().contains("ligne 2"));
    }
}
