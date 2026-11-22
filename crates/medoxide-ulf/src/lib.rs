//! Reconstruction IRM ultra-bas champ (48 mT), étape 1 : modèle direct d'une acquisition avec champ B0, et son adjoint
//! (la reconstruction par phase conjuguée, CPR). Portage de `fmi.reconstruction.operators.cpr` ; critères et résultats :
//! docs/ULF.md (P4). Estimation jointe B0 + image (P4 bis) : INR à grille de hachage ([`inr`]).

pub mod cpr;
pub mod inr;

pub use cpr::{ModeleCpr, Reel};
