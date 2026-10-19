//! Normalisation d'instance à **réductions en cascade**, en remplacement de `burn::nn::InstanceNorm` pour les volumes 3D de grande taille.
//!
//! Pourquoi : la normalisation d'instance de Burn (et de PyTorch) calcule, pour chaque canal, la moyenne et la variance sur TOUS les voxels. À 256³ ce sont 16,7 millions de termes sommés en `f32` en un seul passage de réduction : l'erreur d'arrondi croît avec le nombre de termes et
//! sature la précision à grande taille. Mesuré contre un calcul en `f64` sur des tenseurs de moyenne 3 et d'écart type 1 : écart maximal de la sortie normalisée 2,8e-5 à 64³, 4,7e-4 à 128³ et 1,0e-2 à 256³ sur GPU (1,7e-4, 3,3e-3 et 1,6e-1 sur CPU), alors que les valeurs normalisées sont de l'ordre de 1.
//! Ici la moyenne et la variance sont calculées une dimension spatiale après l'autre (trois réductions de 256 termes à 256³ au lieu d'une de 16,7 millions) : la moyenne des moyennes est exacte quand les blocs ont la même taille.
//!
//! Deuxième source d'erreur, indépendante de la taille : quand la moyenne d'un canal est grande devant son écart type, sommer des valeurs de l'ordre de la moyenne perd de la précision relative à l'écart type (mesuré, sortie normalisée contre f64 sur 128³ : 1e-6 pour un rapport moyenne / écart type de 1, 1,2e-5 pour 100, 2,4e-4 pour 1000). Les cartes de caractéristiques du réseau ont de tels rapports.
//! Remède, la moyenne corrigée en deux passes : une première moyenne `m0` recentre les données (la soustraction de valeurs voisines est exacte), puis la moyenne résiduelle, minuscule, se calcule sans cette perte.
//!
//! Même structure, mêmes noms de champs et même configuration que `burn::nn::InstanceNorm` : le code généré par `onnx2burn` et ses poids (`gamma`, `beta`) s'y branchent sans autre changement que l'import.
// avertissement de clippy sur le code que la macro `Config` de Burn génère pour `InstanceNormConfig` (bloc `impl` séparé de la structure)
#![allow(clippy::redundant_field_names)]
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{Device, Tensor};

/// Réglages d'une [`InstanceNorm`] (mêmes champs que `burn::nn::InstanceNormConfig`).
#[derive(Debug, Config)]
pub struct InstanceNormConfig {
    /// Nombre de canaux attendus en entrée.
    pub num_channels: usize,
    /// Terme de stabilité numérique ajouté à la variance (1e-5 par défaut).
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// Si `true`, paramètres affines par canal (`gamma` initialisé à 1, `beta` à 0).
    #[config(default = true)]
    pub affine: bool,
}

/// Normalisation d'instance : pour chaque exemple et chaque canal, `(x − moyenne) / √(variance + ε)` sur les dimensions spatiales, puis `gamma · . + beta` si `affine`.
#[derive(Module, Debug)]
pub struct InstanceNorm {
    /// Poids par canal (si `affine`).
    pub gamma: Option<Param<Tensor<1>>>,
    /// Biais par canal (si `affine`).
    pub beta: Option<Param<Tensor<1>>>,
    /// Nombre de canaux attendus en entrée.
    pub num_channels: usize,
    /// Terme de stabilité numérique ajouté à la variance.
    pub epsilon: f64,
    /// Paramètres affines présents ou non.
    pub affine: bool,
}

impl InstanceNormConfig {
    /// Construit la normalisation sur `device`.
    pub fn init(&self, device: &Device) -> InstanceNorm {
        let (gamma, beta) = if self.affine {
            (
                Some(Initializer::Ones.init([self.num_channels], device)),
                Some(Initializer::Zeros.init([self.num_channels], device)),
            )
        } else {
            (None, None)
        };
        InstanceNorm {
            gamma,
            beta,
            num_channels: self.num_channels,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

/// Moyenne sur les dimensions spatiales `2..D` d'un tenseur `[exemples, canaux, ...]`, une dimension à la fois (la dernière d'abord) ; la forme garde des dimensions de taille 1.
fn moyenne_en_cascade<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    (2..D).rev().fold(x, |m, dimension| m.mean_dim(dimension))
}

impl InstanceNorm {
    /// Passe avant sur une entrée `[exemples, canaux, dimensions spatiales…]`.
    pub fn forward<const D: usize>(&self, entree: Tensor<D>) -> Tensor<D> {
        let approchee = moyenne_en_cascade(entree.clone());
        let decalee = entree.sub(approchee);
        let residuelle = moyenne_en_cascade(decalee.clone());
        let centree = decalee.sub(residuelle);
        let variance = moyenne_en_cascade(centree.clone().square());
        let normalisee = centree.div(variance.add_scalar(self.epsilon).sqrt());
        match (&self.gamma, &self.beta) {
            (Some(gamma), Some(beta)) => {
                let mut forme = [1usize; D];
                forme[1] = self.num_channels;
                normalisee
                    .mul(gamma.val().reshape(forme))
                    .add(beta.val().reshape(forme))
            }
            _ => normalisee,
        }
    }
}
