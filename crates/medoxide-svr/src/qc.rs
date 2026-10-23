//! **Contrôle qualité des acquisitions, avant reconstruction** : pour chaque coupe de chaque pile, ce qu'on peut savoir sans recaler ni reconstruire. Écrit `…_desc-acq_slicequality.tsv` (voir [`ecrire`]), au format du fichier de qualité du SVR avec trois colonnes de plus.
//!
//! Indicateurs par coupe :
//! - `mask_pixels` : nombre de pixels de masque cérébral ;
//! - `mean_signal` : intensité moyenne dans le masque, et son rapport au plus faible des deux signaux des coupes adjacentes du même stack (un **creux** : une coupe sombre ou vidée est sous ses deux voisines, alors que le signal baisse sans creux vers les bords du cerveau) ;
//! - `n_partners` : nombre de coupes d'autres piles qui croisent la coupe aux poses d'en-tête ;
//! - `score_before` : concordance aux intersections aux poses d'en-tête, avant tout recalage (mesure du mouvement à corriger).
//!
//! Une coupe est **signalée** (`excluded` 1, motif dans `reason`) pour l'un de ces motifs, dans cet ordre : `hors_masque`, `masque_insuffisant`, `sans_partenaire` (le SVR ne la recalera pas), `signal_faible` (rapport < [`RAPPORT_SIGNAL_MIN`]). Le contrôle informe seulement : le SVR n'en tient pas compte.

use std::path::Path;

use burn::prelude::*;
use medoxide_core::bids::{self, Generateur};
use medoxide_core::json::Json;
use nalgebra::Matrix4;
use ndarray::Axis;
use rayon::prelude::*;

use crate::align::{preparer_coupe, MIN_MASK_PIXELS};
use crate::pipeline::stack_names;
use crate::recon::SlicePoses;
use crate::s2s::{IntersectionProblem, S2sConfig, SliceAtlas};
use crate::{Stack, SvrError};

/// Rapport minimal du signal moyen d'une coupe au plus faible des signaux de ses deux coupes adjacentes : en dessous, la coupe est un creux et est signalée `signal_faible`.
/// Première version (écartée après la première exécution) : médiane des coupes à ±2 ; elle signalait 7 coupes sur 36 d'un jeu sans défaut, parce que le signal chute vers les bords du cerveau.
pub const RAPPORT_SIGNAL_MIN: f64 = 0.6;

/// Motif pour lequel une coupe est signalée.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotifQc {
    /// Aucun pixel de masque : la coupe ne contribue pas.
    HorsMasque,
    /// Moins de [`MIN_MASK_PIXELS`] pixels de masque : le SVR ne la recale pas.
    MasqueInsuffisant,
    /// Masque suffisant mais aucune coupe d'une autre pile ne la croise : le SVR ne la recale pas.
    SansPartenaire,
    /// Creux de signal : signal moyen inférieur à [`RAPPORT_SIGNAL_MIN`] fois celui du plus faible des deux coupes adjacentes.
    SignalFaible,
}

impl MotifQc {
    /// Nom écrit dans la colonne `reason`.
    pub fn nom(self) -> &'static str {
        match self {
            MotifQc::HorsMasque => "hors_masque",
            MotifQc::MasqueInsuffisant => "masque_insuffisant",
            MotifQc::SansPartenaire => "sans_partenaire",
            MotifQc::SignalFaible => "signal_faible",
        }
    }
}

/// Une ligne du contrôle : une coupe.
#[derive(Debug, Clone, PartialEq)]
pub struct LigneQc {
    /// Nom de la pile (sans extension).
    pub stack: String,
    /// Indice de la coupe dans la pile.
    pub slice: usize,
    /// Pixels de masque.
    pub mask_pixels: usize,
    /// Intensité moyenne dans le masque ; `None` sans pixel de masque.
    pub mean_signal: Option<f64>,
    /// Rapport du signal moyen au plus faible des deux coupes adjacentes ; `None` si l'une des deux n'existe pas ou n'est pas exploitable (masque insuffisant), ou sans signal.
    pub rapport_signal: Option<f64>,
    /// Coupes d'autres piles qui croisent celle-ci aux poses d'en-tête.
    pub n_partners: usize,
    /// Concordance aux poses d'en-tête (avant recalage) ; `None` sans partenaire ou masque insuffisant.
    pub score_before: Option<f64>,
    /// Motif de signalement, s'il y en a un.
    pub motif: Option<MotifQc>,
}

/// **Contrôle qualité** des acquisitions `stacks` (avec masques) : une ligne par coupe, dans l'ordre des piles puis des coupes. `config.intersection` règle les paires (comme le SVR).
///
/// # Erreurs
/// [`SvrError::NoMask`] si une pile n'a pas de masque ; [`SvrError::DuplicateStackName`] si deux piles ont le même nom.
pub fn qualite_acquisition(stacks: &[Stack], config: &S2sConfig, device: &Device) -> Result<Vec<LigneQc>, SvrError> {
    let poses = SlicePoses::identity(stacks);
    crate::align::verifier_entrees(stacks, &poses)?;
    let noms = stack_names(stacks)?;
    // pixels de masque et signal moyen de chaque coupe
    let mut lignes: Vec<LigneQc> = Vec::new();
    for (si, stack) in stacks.iter().enumerate() {
        let masque = stack.brain_mask().expect("vérifié").voxels();
        for k in 0..stack.dim().2 {
            let plan = masque.index_axis(Axis(2), k);
            let donnees = stack.data().index_axis(Axis(2), k);
            let (mut n, mut somme) = (0usize, 0.0_f64);
            for (m, v) in plan.iter().zip(donnees.iter()) {
                if *m {
                    n += 1;
                    somme += f64::from(*v);
                }
            }
            lignes.push(LigneQc {
                stack: noms[si].clone(),
                slice: k,
                mask_pixels: n,
                mean_signal: (n > 0).then(|| somme / n as f64),
                rapport_signal: None,
                n_partners: 0,
                score_before: None,
                motif: None,
            });
        }
    }
    // creux de signal : rapport au plus faible des deux coupes adjacentes (toutes deux exploitables) du même stack
    let mut debut = 0;
    for stack in stacks {
        let n = stack.dim().2;
        let bloc = &mut lignes[debut..debut + n];
        let signaux: Vec<Option<f64>> = bloc
            .iter()
            .map(|l| (l.mask_pixels >= MIN_MASK_PIXELS).then_some(l.mean_signal).flatten())
            .collect();
        for k in 1..n.saturating_sub(1) {
            if let (Some(propre), Some(avant), Some(apres)) = (bloc[k].mean_signal, signaux[k - 1], signaux[k + 1]) {
                let faible = avant.min(apres);
                if faible > 0.0 {
                    bloc[k].rapport_signal = Some(propre / faible);
                }
            }
        }
        debut += n;
    }
    // partenaires et concordance aux poses d'en-tête (en parallèle : lecture seule)
    let atlas = SliceAtlas::new(stacks, device)?;
    let coupes: Vec<(usize, usize)> = stacks
        .iter()
        .enumerate()
        .flat_map(|(si, s)| (0..s.dim().2).map(move |k| (si, k)))
        .collect();
    let mesures: Vec<(usize, Option<f64>)> = coupes
        .par_iter()
        .map(|&(si, k)| {
            let identite = Matrix4::identity();
            let Some(preparee) = preparer_coupe(&stacks[si], k, identite) else {
                return (0, None);
            };
            let Some(probleme) =
                IntersectionProblem::build(&atlas, stacks, &poses, si, k, &config.intersection, device)
            else {
                return (0, None);
            };
            let pivot = Tensor::<1>::from_floats(
                [
                    preparee.pivot.x as f32,
                    preparee.pivot.y as f32,
                    preparee.pivot.z as f32,
                ],
                device,
            );
            let zero = Tensor::<1>::from_floats([0.0_f32; 6], device);
            let v = probleme.cost(zero, pivot, preparee.echelle);
            (
                probleme.n_pairs(),
                Some(f64::from(v.into_data().try_to_vec::<f32>().unwrap()[0])),
            )
        })
        .collect();
    for (ligne, (partenaires, score)) in lignes.iter_mut().zip(mesures) {
        ligne.n_partners = partenaires;
        ligne.score_before = score;
        ligne.motif = if ligne.mask_pixels == 0 {
            Some(MotifQc::HorsMasque)
        } else if ligne.mask_pixels < MIN_MASK_PIXELS {
            Some(MotifQc::MasqueInsuffisant)
        } else if partenaires == 0 {
            Some(MotifQc::SansPartenaire)
        } else if ligne.rapport_signal.is_some_and(|r| r < RAPPORT_SIGNAL_MIN) {
            Some(MotifQc::SignalFaible)
        } else {
            None
        };
    }
    Ok(lignes)
}

/// Texte du fichier `…_desc-acq_slicequality.tsv` : l'en-tête et les colonnes du fichier de qualité du SVR (`registered` vaut 0 : rien n'est recalé ; `score_after` et `correction_rms_mm` sont absents, `n/a`), puis `mask_pixels`, `mean_signal` et `n_partners`. Terminé par un saut de ligne.
pub fn en_tsv(lignes: &[LigneQc]) -> String {
    let reel = |v: Option<f64>| {
        v.filter(|x| x.is_finite())
            .map_or_else(|| "n/a".to_string(), |x| format!("{x:?}"))
    };
    let mut sortie = String::from(
        "stack\tslice\tregistered\tscore_before\tscore_after\tn_registrations\tcorrection_rms_mm\texcluded\treason\tmask_pixels\tmean_signal\tn_partners\n",
    );
    for l in lignes {
        sortie.push_str(&format!(
            "{}\t{}\t0\t{}\tn/a\t0\tn/a\t{}\t{}\t{}\t{}\t{}\n",
            l.stack,
            l.slice,
            reel(l.score_before),
            u8::from(l.motif.is_some()),
            l.motif.map_or("n/a", MotifQc::nom),
            l.mask_pixels,
            reel(l.mean_signal),
            l.n_partners
        ));
    }
    sortie
}

/// Fichier voisin du TSV (`…_desc-acq_slicequality.json`) : sens des colonnes et des motifs.
fn description_du_controle(generateur: &Generateur) -> Json {
    let mut champs = vec![
        (
            "Description",
            Json::texte("Contrôle qualité des acquisitions avant reconstruction : une ligne par coupe de chaque pile"),
        ),
        (
            "Colonnes",
            Json::objet([
                ("registered", Json::texte("toujours 0 : rien n'est recalé à ce stade")),
                ("score_before", Json::texte("concordance aux intersections aux poses d'en-tête (−1 à 1, plus grand = mieux) ; n/a sans partenaire ou masque insuffisant")),
                ("excluded", Json::texte("1 si la coupe est signalée : à écarter ou à vérifier (le SVR n'en tient pas compte) ; pour hors_masque, masque_insuffisant et sans_partenaire, le SVR ne la recalera pas")),
                ("reason", Json::texte("hors_masque, masque_insuffisant, sans_partenaire ou signal_faible ; n/a sinon")),
                ("mask_pixels", Json::texte("pixels de masque cérébral de la coupe")),
                ("mean_signal", Json::texte("intensité moyenne dans le masque")),
                ("n_partners", Json::texte("coupes d'autres piles qui croisent la coupe aux poses d'en-tête")),
            ]),
        ),
        (
            "Seuils",
            Json::objet([
                ("masque_insuffisant_pixels", Json::Entier(MIN_MASK_PIXELS as i64)),
                ("signal_faible_rapport_au_plus_faible_des_deux_coupes_adjacentes", Json::Nombre(RAPPORT_SIGNAL_MIN)),
            ]),
        ),
    ];
    if let Some(ligne) = &generateur.ligne_de_commande {
        champs.push(("CommandLine", Json::texte(ligne)));
    }
    Json::objet(champs)
}

/// Écrit le contrôle : `tsv` (dossiers créés au besoin), son fichier voisin `.json` (même nom, extension `json`), et, si `derive` est donné (mode BIDS), `dataset_description.json` du pipeline `medx-svr`.
///
/// # Erreurs
/// [`SvrError::Bids`] si un fichier ne peut pas être écrit.
pub fn ecrire(lignes: &[LigneQc], tsv: &Path, derive: Option<&Path>, generateur: &Generateur) -> Result<(), SvrError> {
    bids::ecrire_texte(tsv, &en_tsv(lignes))?;
    bids::ecrire_texte(
        &tsv.with_extension("json"),
        &description_du_controle(generateur).en_texte(),
    )?;
    if let Some(derive) = derive {
        bids::ecrire_description_derivee(derive, bids::PIPELINE_SVR, generateur)?;
    }
    Ok(())
}
