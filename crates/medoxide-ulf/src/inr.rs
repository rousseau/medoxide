//! Représentation neuronale implicite (INR) à grille de hachage multi-résolution suivie d'un MLP, et ses deux usages dans
//! l'estimation jointe : le champ B0 et l'image complexe. Portage de fmi `notebooks/francois/inr.py` (`HashGridINR`,
//! équivalent PyTorch de tiny-cuda-nn) et de `utils_cpr.py` (`INRField`, `INRImage`), référence `cb7174b` ; critères et
//! résultats : docs/ULF.md (P4 bis, critère 1).
//!
//! Les points évalués sont toujours les voxels d'une même grille : les indices de table et les poids trilinéaires de
//! chaque niveau se calculent donc une seule fois, sur CPU, en `f64` et en entiers 64 bits ([`Encodage`]), avec
//! l'arithmétique de PyTorch (produits masqués sur 32 bits, XOR, modulo la taille de la table ; niveaux denses
//! indexés par pas, même modulo au bord). L'encodage devient une sélection dans la table pondérée par ces poids, que
//! l'autodiff de Burn dérive (accumulation dans la table).

use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{DType, Distribution, TensorData};

/// Nombres premiers du hachage spatial de tiny-cuda-nn (un par axe ; le premier axe n'est pas mélangé).
const PREMIERS: [i64; 3] = [1, 2_654_435_761, 805_459_861];

/// Réglage d'une grille de hachage multi-résolution (noms de tiny-cuda-nn entre parenthèses).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfigGrille {
    /// Nombre de niveaux de résolution (`n_levels`).
    pub n_niveaux: usize,
    /// Caractéristiques par sommet et par niveau (`n_features_per_level`).
    pub n_caracteristiques: usize,
    /// log2 de la taille maximale de la table d'un niveau (`log2_hashmap_size`).
    pub log2_table: u32,
    /// Résolution du niveau le plus grossier (`base_resolution`).
    pub resolution_base: f64,
    /// Facteur géométrique entre deux niveaux (`per_level_scale`).
    pub facteur: f64,
}

impl ConfigGrille {
    /// Grille du champ B0 de P1 (`compare_methods.B0_INR_CONFIG`, réglage finest32) : 8 niveaux, résolution 2 → 32.
    pub fn champ_p1() -> Self {
        Self {
            n_niveaux: 8,
            n_caracteristiques: 2,
            log2_table: 19,
            resolution_base: 2.0,
            facteur: 16f64.powf(1.0 / 7.0),
        }
    }

    /// Grille de l'image de P1 (`compare_methods.U_INR_CONFIG`) : 16 niveaux, résolution de base 16, facteur 2.
    pub fn image_p1() -> Self {
        Self {
            n_niveaux: 16,
            n_caracteristiques: 2,
            log2_table: 19,
            resolution_base: 16.0,
            facteur: 2.0,
        }
    }

    /// Géométrie du niveau `l` (`_HashGridLevel` de fmi).
    pub fn niveau(&self, l: usize) -> Niveau {
        let echelle = self.resolution_base * self.facteur.powf(l as f64) - 1.0;
        let resolution = echelle.ceil() as i64 + 1;
        let n_sommets = resolution.pow(3);
        let taille_table = 8 * (n_sommets.min(1 << self.log2_table) as usize).div_ceil(8);
        Niveau {
            echelle,
            resolution,
            taille_table,
            hachee: n_sommets > taille_table as i64,
        }
    }

    /// Nombre de caractéristiques en sortie de l'encodage (entrée du MLP).
    pub fn n_sorties(&self) -> usize {
        self.n_niveaux * self.n_caracteristiques
    }
}

/// Géométrie d'un niveau de la grille.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Niveau {
    /// Échelle des coordonnées : un point x ∈ [0, 1] tombe en `x · echelle + 0,5` sur le réseau de sommets.
    pub echelle: f64,
    /// Sommets par axe, `⌈echelle⌉ + 1`.
    pub resolution: i64,
    /// Lignes de la table (arrondies au multiple de 8 supérieur, comme tiny-cuda-nn).
    pub taille_table: usize,
    /// Vrai si les sommets ne tiennent pas dans la table (indices hachés) ; sinon indexation dense.
    pub hachee: bool,
}

/// Coordonnées normalisées d'un axe de n voxels, identiques bit pour bit à `torch.linspace(0, 1, n)` en float64 : la
/// première moitié part de 0, la seconde de 1, chaque valeur en une multiplication-addition fusionnée (un seul
/// arrondi). Vérifié pour n = 2 à 299 contre PyTorch sur CPU arm64 (Mac), CPU x86_64 et CUDA (RTX) ; sans fusion,
/// 3 104 valeurs sur ces tailles diffèrent d'un ulp (n = 45 : indices 26, 30, 41).
pub fn linspace(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![0.0];
    }
    let pas = 1.0 / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i < n / 2 {
                pas * i as f64
            } else {
                (-pas).mul_add((n - 1 - i) as f64, 1.0)
            }
        })
        .collect()
}

/// Indices de table et poids trilinéaires des 8 coins de chaque voxel d'une grille `dims` (ordre ligne par ligne,
/// premier axe le plus lent ; coins dans l'ordre de fmi : le bit d du numéro de coin choisit le sommet suivant le long
/// de l'axe d). Renvoie deux vecteurs de 8 · n_points valeurs.
pub fn indices_et_poids(niveau: &Niveau, dims: [usize; 3]) -> (Vec<i64>, Vec<f64>) {
    let axes: Vec<Vec<f64>> = dims.iter().map(|&n| linspace(n)).collect();
    let n = dims.iter().product::<usize>();
    let (mut indices, mut poids) = (Vec::with_capacity(8 * n), Vec::with_capacity(8 * n));
    let taille = niveau.taille_table as i64;
    for &c0 in &axes[0] {
        for &c1 in &axes[1] {
            for &c2 in &axes[2] {
                let mut base = [0i64; 3];
                let mut frac = [0f64; 3];
                for (d, c) in [c0, c1, c2].into_iter().enumerate() {
                    let x = c * niveau.echelle + 0.5;
                    let xi = x.floor();
                    base[d] = xi as i64;
                    frac[d] = x - xi;
                }
                for coin in 0..8 {
                    let mut sommet = [0i64; 3];
                    let mut w = 1.0;
                    for d in 0..3 {
                        let suivant = coin & (1 << d) != 0;
                        sommet[d] = base[d] + suivant as i64;
                        w *= if suivant { frac[d] } else { 1.0 - frac[d] };
                    }
                    let id = if niveau.hachee {
                        let h = (0..3).fold(0i64, |h, d| h ^ ((sommet[d] * PREMIERS[d]) & 0xFFFF_FFFF));
                        h % taille
                    } else {
                        let r = niveau.resolution;
                        (sommet[0] + sommet[1] * r + sommet[2] * r * r) % taille
                    };
                    indices.push(id);
                    poids.push(w);
                }
            }
        }
    }
    (indices, poids)
}

/// Indices et poids de tous les niveaux pour une grille donnée, sur le device de calcul (poids dans le type des
/// réseaux qui l'utiliseront). Les points sont complétés jusqu'à un multiple de [`BLOC`] par des points fictifs
/// (indice 0, poids nuls), pour que le réseau tourne par blocs (voir [`InrHachage::evaluer`]).
#[derive(Debug, Clone)]
pub struct Encodage {
    /// Dimensions de la grille.
    pub dims: [usize; 3],
    /// Par niveau : indices (8 · points complétés) et poids (8 · points complétés, 1).
    niveaux: Vec<(Tensor<1, Int>, Tensor<2>)>,
}

impl Encodage {
    /// Précalcule l'encodage de la grille `dims` pour la configuration `config`.
    pub fn nouveau(config: &ConfigGrille, dims: [usize; 3], device: &Device, dtype: DType) -> Self {
        let complet = 8 * dims.iter().product::<usize>().div_ceil(BLOC) * BLOC;
        let niveaux = (0..config.n_niveaux)
            .map(|l| {
                let (mut indices, mut poids) = indices_et_poids(&config.niveau(l), dims);
                indices.resize(complet, 0);
                poids.resize(complet, 0.0);
                (
                    Tensor::from_data(TensorData::new(indices, [complet]), device),
                    Tensor::from_data(TensorData::new(poids, [complet, 1]), (device, dtype)),
                )
            })
            .collect();
        Self { dims, niveaux }
    }

    /// Nombre de points (voxels).
    pub fn n_points(&self) -> usize {
        self.dims.iter().product()
    }

    /// Nombre de blocs de [`BLOC`] points.
    fn n_blocs(&self) -> usize {
        self.n_points().div_ceil(BLOC)
    }
}

/// Grille de hachage multi-résolution suivie d'un MLP (ReLU entre les couches, sortie linéaire).
#[derive(Module, Debug)]
pub struct InrHachage {
    /// Une table par niveau, forme (taille de table, caractéristiques).
    pub tables: Vec<Param<Tensor<2>>>,
    /// Couches du MLP ; poids de forme (entrées, sorties).
    pub couches: Vec<Linear>,
}

fn uniforme<const D: usize>(forme: [usize; D], borne: f64, device: &Device, dtype: DType) -> Tensor<D> {
    Tensor::<D>::zeros(forme, (device, dtype)).random_like(Distribution::Uniform(-borne, borne))
}

impl InrHachage {
    /// Réseau initialisé comme dans fmi : tables uniformes dans ± 1e-4, couches linéaires (poids et biais) uniformes
    /// dans ± 1/√(entrées), comme `nn.Linear`.
    pub fn nouveau(
        config: &ConfigGrille,
        n_sorties: usize,
        n_neurones: usize,
        n_cachees: usize,
        device: &Device,
        dtype: DType,
    ) -> Self {
        let tables = (0..config.n_niveaux)
            .map(|l| {
                let forme = [config.niveau(l).taille_table, config.n_caracteristiques];
                Param::from_tensor(uniforme(forme, 1e-4, device, dtype))
            })
            .collect();
        let mut largeurs = vec![config.n_sorties()];
        largeurs.extend(std::iter::repeat_n(n_neurones, n_cachees));
        largeurs.push(n_sorties);
        let couches = largeurs
            .windows(2)
            .map(|w| {
                let borne = 1.0 / (w[0] as f64).sqrt();
                Linear {
                    weight: Param::from_tensor(uniforme([w[0], w[1]], borne, device, dtype)),
                    bias: Some(Param::from_tensor(uniforme([w[1]], borne, device, dtype))),
                }
            })
            .collect();
        Self { tables, couches }
    }

    /// Réseau aux poids donnés : tables (valeurs ligne par ligne, forme) et couches (poids au format PyTorch, forme
    /// (sorties, entrées), puis biais).
    pub fn depuis_poids(
        tables: Vec<(Vec<f64>, [usize; 2])>,
        couches: Vec<(Vec<f64>, [usize; 2], Vec<f64>)>,
        device: &Device,
        dtype: DType,
    ) -> Self {
        let opts = (device, dtype);
        let tables = tables
            .into_iter()
            .map(|(v, f)| Param::from_tensor(Tensor::from_data(TensorData::new(v, f), opts)))
            .collect();
        let couches = couches
            .into_iter()
            .map(|(w, f, b)| {
                let n = b.len();
                Linear {
                    weight: Param::from_tensor(Tensor::<2>::from_data(TensorData::new(w, f), opts).transpose()),
                    bias: Some(Param::from_tensor(Tensor::from_data(TensorData::new(b, [n]), opts))),
                }
            })
            .collect();
        Self { tables, couches }
    }

    /// Encodage du niveau `l` par blocs, forme (blocs, BLOC, caractéristiques), points fictifs compris.
    fn encodage_blocs(&self, encodage: &Encodage, l: usize) -> Tensor<3> {
        let (indices, poids) = &encodage.niveaux[l];
        let table = self.tables[l].val();
        let n_carac = table.dims()[1];
        let k = encodage.n_blocs();
        (table.select(0, indices.clone()) * poids.clone())
            .reshape([k * BLOC, 8, n_carac])
            .sum_dim(1)
            .reshape([k, BLOC, n_carac])
    }

    /// Sortie de l'encodage du niveau `l`, forme (points, caractéristiques).
    pub fn encodage_niveau(&self, encodage: &Encodage, l: usize) -> Tensor<2> {
        let x = self.encodage_blocs(encodage, l);
        let [k, _, c] = x.dims();
        x.reshape([k * BLOC, c]).slice([0..encodage.n_points(), 0..c])
    }

    /// Sortie brute du réseau sur la grille de l'encodage, forme (points, sorties).
    ///
    /// Le MLP tourne par blocs de [`BLOC`] points (produit matriciel par lots, biais diffusé en deux étapes) : au
    /// retour, les gradients des poids et des biais, sommes sur tous les points, se font ainsi en deux niveaux
    /// (dans un bloc, puis entre blocs) au lieu d'une somme séquentielle de n termes (voir [`diffuser`]).
    pub fn evaluer(&self, encodage: &Encodage) -> Tensor<2> {
        let niveaux = (0..self.tables.len())
            .map(|l| self.encodage_blocs(encodage, l))
            .collect();
        let mut x = Tensor::cat(niveaux, 2);
        let k = encodage.n_blocs();
        for (i, couche) in self.couches.iter().enumerate() {
            let b = couche.bias.as_ref().expect("couches avec biais").val();
            let n_sorties = b.dims()[0];
            let b = b
                .reshape([1, 1, n_sorties])
                .expand([k, 1, n_sorties])
                .expand([k, BLOC, n_sorties]);
            x = x.matmul(couche.weight.val().unsqueeze::<3>()) + b;
            if i + 1 < self.couches.len() {
                x = relu(x);
            }
        }
        let n_sorties = x.dims()[2];
        x.reshape([k * BLOC, n_sorties])
            .slice([0..encodage.n_points(), 0..n_sorties])
    }
}

/// Blocs des sommes à deux niveaux ([`moyenne`], [`diffuser`]).
const BLOC: usize = 1024;

/// Valeur scalaire (forme (1)) répétée sur n points, en deux diffusions successives (1 → blocs → points) : au retour,
/// Burn somme le gradient d'une diffusion en une seule boucle séquentielle par valeur diffusée (CPU comme GPU), soit
/// une erreur en n · ε ; deux diffusions donnent deux sommes courtes. Le gradient de l'offset du champ passait ainsi
/// de 1,0e-5 à 2,5e-7 du float64 en `f32` sur 595 125 voxels (PyTorch : 6,7e-8).
pub fn diffuser(x: Tensor<1>, n: usize) -> Tensor<1> {
    let k = n.div_ceil(BLOC);
    x.expand([k])
        .reshape([k, 1])
        .expand([k, BLOC])
        .reshape([k * BLOC])
        .slice(0..n)
}

/// Moyenne d'un vecteur en deux niveaux de sommes (blocs de 1 024, complétés par des zéros) : l'erreur d'arrondi
/// d'une somme séquentielle de n valeurs croît comme n · ε, ce qui en `f32` sur 595 125 voxels donnait au champ un
/// écart de 8,8e-5 au float64 (PyTorch, qui somme par blocs : 1,4e-6 ; ici 6,9e-7). Forme (1).
pub fn moyenne(x: Tensor<1>) -> Tensor<1> {
    let n = x.dims()[0];
    let complet = n.div_ceil(BLOC) * BLOC;
    let x = if complet > n {
        let zeros = Tensor::<1>::zeros([complet - n], (&x.device(), x.dtype()));
        Tensor::cat(vec![x, zeros], 0)
    } else {
        x
    };
    // Multiplication par 1/n plutôt que division par n : au retour, burn-autodiff 0.22.1 calcule l'inverse d'un
    // diviseur scalaire en f32 (`float_div_scalar`), ce qui limite le gradient à environ 1e-8 relatif, même en f64.
    x.reshape([complet / BLOC, BLOC]).sum_dim(1).sum_dim(0).reshape([1]) * (1.0 / n as f64)
}

/// Champ B0 (Hz) = offset + amplitude · forme (`utils_cpr.INRField`) :
/// - forme = tanh(réseau), centrée et ramenée à une moyenne quadratique de 1 (dénominateur détaché du gradient) ;
/// - amplitude = b0_min · (b0_max / b0_min)^sigmoid(s), bornée dans [b0_min, b0_max] ;
/// - offset = b0_max · tanh(o).
#[derive(Module, Debug)]
pub struct ChampB0 {
    /// Réseau à une sortie.
    pub reseau: InrHachage,
    /// Paramètre de l'amplitude (forme (1)).
    pub s: Param<Tensor<1>>,
    /// Paramètre de l'offset (forme (1)).
    pub o: Param<Tensor<1>>,
    /// Bornes de l'amplitude (Hz).
    pub b0_min_hz: f64,
    pub b0_max_hz: f64,
}

impl ChampB0 {
    /// Champ initialisé comme dans fmi (s = o = 0).
    pub fn nouveau(reseau: InrHachage, b0_min_hz: f64, b0_max_hz: f64) -> Self {
        let t = reseau.tables[0].val();
        let (device, dtype) = (t.device(), t.dtype());
        let zero = || Param::from_tensor(Tensor::<1>::zeros([1], (&device, dtype)));
        Self {
            reseau,
            s: zero(),
            o: zero(),
            b0_min_hz,
            b0_max_hz,
        }
    }

    /// Amplitude (Hz), forme (1).
    pub fn amplitude_hz(&self) -> Tensor<1> {
        (sigmoid(self.s.val()) * (self.b0_max_hz / self.b0_min_hz).ln()).exp() * self.b0_min_hz
    }

    /// Offset (Hz), forme (1).
    pub fn offset_hz(&self) -> Tensor<1> {
        self.o.val().tanh() * self.b0_max_hz
    }

    /// Champ sur la grille de l'encodage, aplati (points).
    pub fn b0(&self, encodage: &Encodage) -> Tensor<1> {
        let forme = self.reseau.evaluer(encodage).reshape([encodage.n_points()]).tanh();
        let n = encodage.n_points();
        let forme = forme.clone() - diffuser(moyenne(forme), n);
        let denominateur = moyenne(forme.clone().powi_scalar(2)).detach();
        let forme = forme / diffuser((denominateur + 1e-6).sqrt(), n);
        diffuser(self.offset_hz(), n) + diffuser(self.amplitude_hz(), n) * forme
    }

    /// Pénalité L2 des deux scalaires, s² + o² (forme (1)).
    pub fn penalite_scalaires(&self) -> Tensor<1> {
        self.s.val().powi_scalar(2) + self.o.val().powi_scalar(2)
    }
}

/// Image complexe = réseau à deux sorties, parties réelle et imaginaire (`utils_cpr.INRImage`).
#[derive(Module, Debug)]
pub struct ImageInr {
    /// Réseau à deux sorties.
    pub reseau: InrHachage,
}

impl ImageInr {
    /// Parties réelle et imaginaire sur la grille de l'encodage, aplaties (points).
    pub fn image(&self, encodage: &Encodage) -> (Tensor<1>, Tensor<1>) {
        let n = encodage.n_points();
        let sortie = self.reseau.evaluer(encodage);
        (
            sortie.clone().slice([0..n, 0..1]).reshape([n]),
            sortie.slice([0..n, 1..2]).reshape([n]),
        )
    }
}
