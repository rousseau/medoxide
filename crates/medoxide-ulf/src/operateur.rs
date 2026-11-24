//! Modèle CPR en tenseurs Burn sur le dispositif des réseaux (docs/ULF.md, P4 ter) : mêmes calculs que
//! [`crate::ModeleCpr`] (voir sa documentation pour les formules et les conventions de gradient), sans transfert vers
//! le CPU.
//!
//! - Somme le long de la lecture : phases exp(∓2iπ b0 tᵢ) et produits par cᵢ(x₀) en tenseurs [paquet, N₀, N₁N₂], par
//!   paquets d'échantillons (mémoire bornée), réduits par `sum_dim`.
//! - FFT 2D centrée normalisée des coupes : deux produits matriciels par les matrices de la DFT centrée
//!   D[a, b] = exp(−2iπ (a − h)(b − h) / N) / √N, h = N / 2 (symétriques et unitaires : l'inverse est le conjugué).
//! - Sans champ ([`OperateurCpr::transformee`], [`OperateurCpr::transformee_inverse`]) : F₀ est aussi un produit par
//!   la DFT centrée (c'est la matrice des cᵢ(x₀)).
//!
//! Les entrées sont détachées : l'opérateur ne passe pas par l'autodiff (gradients écrits, coupure du graphe).

use std::f64::consts::TAU;

use burn::prelude::*;
use burn::tensor::{DType, TensorData};

/// Tenseur complexe : parties réelle et imaginaire.
#[derive(Clone, Debug)]
pub struct Complexe<const D: usize> {
    pub re: Tensor<D>,
    pub im: Tensor<D>,
}

impl<const D: usize> Complexe<D> {
    pub fn new(re: Tensor<D>, im: Tensor<D>) -> Self {
        Self { re, im }
    }

    pub fn mul(&self, o: &Self) -> Self {
        Self {
            re: self.re.clone() * o.re.clone() - self.im.clone() * o.im.clone(),
            im: self.re.clone() * o.im.clone() + self.im.clone() * o.re.clone(),
        }
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: self.im.clone().neg(),
        }
    }

    pub fn add(&self, o: &Self) -> Self {
        Self {
            re: self.re.clone() + o.re.clone(),
            im: self.im.clone() + o.im.clone(),
        }
    }

    pub fn sub(&self, o: &Self) -> Self {
        Self {
            re: self.re.clone() - o.re.clone(),
            im: self.im.clone() - o.im.clone(),
        }
    }

    /// Produit par un tenseur réel (même forme ou diffusable).
    pub fn mul_reel(&self, r: &Tensor<D>) -> Self {
        Self {
            re: self.re.clone() * r.clone(),
            im: self.im.clone() * r.clone(),
        }
    }

    pub fn reshape<const D2: usize>(&self, forme: [usize; D2]) -> Complexe<D2> {
        Complexe {
            re: self.re.clone().reshape(forme),
            im: self.im.clone().reshape(forme),
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            re: self.re.clone().detach(),
            im: self.im.clone().detach(),
        }
    }

    /// Module au carré.
    pub fn norme2(&self) -> Tensor<D> {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

/// Produit matriciel complexe (lots compris).
fn cmatmul<const D: usize>(a: &Complexe<D>, b: &Complexe<D>) -> Complexe<D> {
    Complexe {
        re: a.re.clone().matmul(b.re.clone()) - a.im.clone().matmul(b.im.clone()),
        im: a.re.clone().matmul(b.im.clone()) + a.im.clone().matmul(b.re.clone()),
    }
}

/// Matrice de la DFT centrée normalisée de taille n (phase calculée en entier modulo n, en `f64`).
fn dft_centree(n: usize, device: &Device, dtype: DType) -> Complexe<2> {
    let h = (n / 2) as i64;
    let norme = 1.0 / (n as f64).sqrt();
    let (mut re, mut im) = (Vec::with_capacity(n * n), Vec::with_capacity(n * n));
    for a in 0..n as i64 {
        for b in 0..n as i64 {
            let m = ((a - h) * (b - h)).rem_euclid(n as i64) as f64;
            let (s, c) = (-TAU * m / n as f64).sin_cos();
            re.push(c * norme);
            im.push(s * norme);
        }
    }
    let t = |v: Vec<f64>| Tensor::<2>::from_data(TensorData::new(v, [n, n]), (device, dtype));
    Complexe::new(t(re), t(im))
}

/// Modèle direct et adjoint sur le dispositif, pour une grille `[N₀, N₁, N₂]` (lecture le long de l'axe 0).
#[derive(Clone, Debug)]
pub struct OperateurCpr {
    dims: [usize; 3],
    /// Temps tᵢ (s), forme (N₀).
    temps: Tensor<1>,
    /// DFT centrée le long de chaque axe ; `d[0][i, x₀]` = cᵢ(x₀).
    d: [Complexe<2>; 3],
    /// Nombre d'échantillons (ou de coupes) traités ensemble dans les sommes le long de la lecture.
    paquet: usize,
}

impl OperateurCpr {
    pub fn new(dims: [usize; 3], temps: &[f64], device: &Device, dtype: DType) -> Self {
        assert_eq!(
            temps.len(),
            dims[0],
            "{} temps pour {} échantillons",
            temps.len(),
            dims[0]
        );
        // paquet : environ 16 millions d'éléments par tenseur intermédiaire
        let paquet = (16_000_000 / (dims[0] * dims[1] * dims[2])).clamp(1, dims[0]);
        Self {
            dims,
            temps: Tensor::from_data(TensorData::new(temps.to_vec(), [dims[0]]), (device, dtype)),
            d: dims.map(|n| dft_centree(n, device, dtype)),
            paquet,
        }
    }

    /// Même opérateur, sommes le long de la lecture par paquets de `paquet` échantillons au plus (les grilles des tests
    /// tiennent en un seul paquet par défaut).
    pub fn avec_paquet(mut self, paquet: usize) -> Self {
        assert!(paquet >= 1, "paquet vide");
        self.paquet = paquet.min(self.dims[0]);
        self
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// FFT 2D centrée (ou son inverse) de chaque coupe d'un volume [B, N₁, N₂].
    fn f12(&self, s: &Complexe<3>, inverse: bool) -> Complexe<3> {
        let b = s.re.dims()[0];
        let [_, n1, n2] = self.dims;
        let lot = |m: &Complexe<2>, n: usize| {
            let m = if inverse { m.conj() } else { m.clone() };
            Complexe::new(
                m.re.unsqueeze::<3>().expand([b, n, n]),
                m.im.unsqueeze::<3>().expand([b, n, n]),
            )
        };
        let r = cmatmul(&lot(&self.d[1], n1), s);
        cmatmul(&r, &lot(&self.d[2], n2))
    }

    /// `A(b0) u` (image → k-space), `b0` en Hz.
    pub fn direct(&self, u: &Complexe<3>, b0: &Tensor<3>) -> Complexe<3> {
        let [n0, n1, n2] = self.dims;
        let n12 = n1 * n2;
        let u2 = u.detach().reshape([1, n0, n12]);
        let b2 = b0.clone().detach().reshape([1, n0, n12]);
        let mut paquets = Vec::new();
        for debut in (0..n0).step_by(self.paquet) {
            let fin = (debut + self.paquet).min(n0);
            let m = fin - debut;
            let t = self.temps.clone().slice(debut..fin).reshape([m, 1, 1]);
            let theta = b2.clone() * t * (-TAU);
            let e = Complexe::new(theta.clone().cos(), theta.sin());
            let c = self.d[0].reshape::<2>([n0, n0]);
            let c = Complexe::new(
                c.re.slice([debut..fin, 0..n0]).reshape([m, n0, 1]),
                c.im.slice([debut..fin, 0..n0]).reshape([m, n0, 1]),
            );
            let w = e.mul(&u2).mul(&c);
            paquets.push(Complexe::new(w.re.sum_dim(1), w.im.sum_dim(1)));
        }
        let s = Complexe::new(
            Tensor::cat(paquets.iter().map(|p| p.re.clone()).collect(), 0),
            Tensor::cat(paquets.iter().map(|p| p.im.clone()).collect(), 0),
        );
        self.f12(&s.reshape([n0, n1, n2]), false)
    }

    /// `Aᴴ(b0) k` (k-space → image).
    pub fn adjoint(&self, k: &Complexe<3>, b0: &Tensor<3>) -> Complexe<3> {
        self.adjoints(k, b0, false).0
    }

    /// `Aᴴ(b0) k` et l'adjoint pondéré par le temps `Aₜᴴ(b0) k`.
    pub fn adjoint_et_pondere(&self, k: &Complexe<3>, b0: &Tensor<3>) -> (Complexe<3>, Complexe<3>) {
        let (a, at) = self.adjoints(k, b0, true);
        (a, at.expect("adjoint pondéré demandé"))
    }

    fn adjoints(&self, k: &Complexe<3>, b0: &Tensor<3>, pondere: bool) -> (Complexe<3>, Option<Complexe<3>>) {
        let [n0, n1, n2] = self.dims;
        let n12 = n1 * n2;
        let p = self.f12(&k.detach(), true).reshape([1, n0, n12]);
        let b2 = b0.clone().detach().reshape([n0, 1, n12]);
        let t = self.temps.clone().reshape([1, n0, 1]);
        // conj(cᵢ(x₀)) rangé [x₀, i] (la DFT centrée est symétrique : c'est conj(D₀) lui-même)
        let cc = self.d[0].conj();
        let (mut s, mut st) = (Vec::new(), Vec::new());
        for debut in (0..n0).step_by(self.paquet) {
            let fin = (debut + self.paquet).min(n0);
            let m = fin - debut;
            let theta = b2.clone().slice([debut..fin, 0..1, 0..n12]) * t.clone() * TAU;
            let e = Complexe::new(theta.clone().cos(), theta.sin());
            let c = Complexe::new(
                cc.re.clone().slice([debut..fin, 0..n0]).reshape([m, n0, 1]),
                cc.im.clone().slice([debut..fin, 0..n0]).reshape([m, n0, 1]),
            );
            let v = e.mul(&p).mul(&c);
            s.push(Complexe::new(v.re.clone().sum_dim(1), v.im.clone().sum_dim(1)));
            if pondere {
                let vt = v.mul_reel(&t);
                st.push(Complexe::new(vt.re.sum_dim(1), vt.im.sum_dim(1)));
            }
        }
        let rassembler = |l: Vec<Complexe<3>>| {
            Complexe::new(
                Tensor::cat(l.iter().map(|p| p.re.clone()).collect(), 0),
                Tensor::cat(l.iter().map(|p| p.im.clone()).collect(), 0),
            )
            .reshape([n0, n1, n2])
        };
        let st = pondere.then(|| rassembler(st));
        (rassembler(s), st)
    }

    /// `A(0) u` = F u (FFT 3D centrée normalisée).
    pub fn transformee(&self, u: &Complexe<3>) -> Complexe<3> {
        let [n0, n1, n2] = self.dims;
        let u2 = u.detach().reshape([n0, n1 * n2]);
        self.f12(&cmatmul(&self.d[0], &u2).reshape([n0, n1, n2]), false)
    }

    /// `Aᴴ(0) k` = F⁻¹ k.
    pub fn transformee_inverse(&self, k: &Complexe<3>) -> Complexe<3> {
        let [n0, n1, n2] = self.dims;
        let p = self.f12(&k.detach(), true).reshape([n0, n1 * n2]);
        cmatmul(&self.d[0].conj(), &p).reshape([n0, n1, n2])
    }

    /// Gradients de L pour y = A(b0) u, connaissant g = ∂L/∂y : (∂L/∂u, ∂L/∂b0).
    pub fn gradient_direct(&self, u: &Complexe<3>, b0: &Tensor<3>, g: &Complexe<3>) -> (Complexe<3>, Tensor<3>) {
        let (ahg, athg) = self.adjoint_et_pondere(g, b0);
        // 2π Im(u · conj(a)) = 2π (u.im a.re − u.re a.im)
        let u = u.detach();
        let gb0 = (u.im * athg.re - u.re * athg.im) * TAU;
        (ahg, gb0)
    }

    /// Gradient de L par rapport à b0 pour z = Aᴴ(b0) k, connaissant g = ∂L/∂z et `atk` = Aₜᴴ(b0) k :
    /// −2π Im(conj(g) · Aₜᴴ k) = −2π (g.re atk.im − g.im atk.re).
    pub fn gradient_adjoint_b0(&self, atk: &Complexe<3>, g: &Complexe<3>) -> Tensor<3> {
        (g.re.clone() * atk.im.clone() - g.im.clone() * atk.re.clone()) * (-TAU)
    }
}
