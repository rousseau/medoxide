//! Reconstruction IRM ultra-bas champ (48 mT), étape 1 : modèle direct d'une acquisition avec champ B0, et son adjoint
//! (la reconstruction par phase conjuguée, CPR). Portage de `fmi.reconstruction.operators.cpr` ; critères et résultats :
//! docs/ULF.md (P4).

pub mod cpr;

pub use cpr::{ModeleCpr, Reel};
