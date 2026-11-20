//! Modèle direct `A(b0)` (image → k-space) et adjoint `Aᴴ(b0)` (k-space → image) d'une acquisition cartésienne 3D dont
//! la lecture suit l'axe 0, avec un champ B0 (Hz) qui déphase chaque échantillon de lecture selon son temps tᵢ :
//!
//! - `A(b0) u [i, :, :] = F(u · exp(−2iπ b0 tᵢ)) [i, :, :]` ;
//! - `Aᴴ(b0) k = Σᵢ F⁻¹(Mᵢ k) · exp(+2iπ b0 tᵢ)`, Mᵢ ne gardant que la ligne i ;
//!
//! où F est la FFT 3D centrée (`ifftshift`, `fftn`, `fftshift`) normalisée (÷ √N), comme dans fmi. Dans fmi, `A` est
//! `time_segmented_cpr_adjoint` (`utils_cpr.predict_kspace`) et `Aᴴ` est `time_segmented_cpr` (`correct_image`).
//!
//! **Calcul.** F se factorise en une transformée 1D le long de la lecture (F₀) et une 2D le long des axes de phase
//! (F₁₂). Comme on ne garde que la ligne i de F₀, une seule ligne de sa matrice sert : cᵢ(x₀) =
//! exp(−2iπ (i − h)(x₀ − h) / N₀) / √N₀, avec h = N₀ / 2 (division entière), qui est la FFT centrée écrite comme une
//! somme. D'où, F₁₂ étant linéaire et appliquée coupe par coupe :
//!
//! - `A(b0) u [i] = F₁₂( Σ_{x₀} cᵢ(x₀) · u[x₀] · exp(−2iπ b0[x₀] tᵢ) )` : **une** FFT 2D par échantillon de lecture ;
//! - `Aᴴ(b0) k [x₀] = Σᵢ conj(cᵢ(x₀)) · exp(+2iπ b0[x₀] tᵢ) · F₁₂⁻¹(k[i])`.
//!
//! fmi fait une FFT 3D complète par échantillon de lecture ; le résultat est le même (vérifié par les tests de parité).
//! Les sommes sont faites dans un ordre fixe : le résultat ne dépend pas du nombre de fils.

use std::sync::Arc;

use ndarray::{Array3, ArrayView2, Axis};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Float, FloatConst};
use rustfft::{Fft, FftNum, FftPlanner};

/// Type flottant des calculs (`f32` ou `f64`) : ce que demandent rustfft (`FftNum`) et les fonctions mathématiques
/// (`Float` : `sqrt`, `sin_cos` ; `FloatConst` : `PI`).
pub trait Reel: FftNum + Float + FloatConst {}

impl<T: FftNum + Float + FloatConst> Reel for T {}

/// FFT 2D centrée et normalisée (÷ √(N₁N₂)) d'une coupe `[N₁, N₂]` rangée ligne par ligne.
struct Fft2Centree<T: Reel> {
    n1: usize,
    n2: usize,
    directe: [Arc<dyn Fft<T>>; 2],
    inverse: [Arc<dyn Fft<T>>; 2],
    echelle: T,
}

impl<T: Reel> Fft2Centree<T> {
    fn new(n1: usize, n2: usize) -> Self {
        let mut planif = FftPlanner::<T>::new();
        Fft2Centree {
            n1,
            n2,
            directe: [planif.plan_fft_forward(n1), planif.plan_fft_forward(n2)],
            inverse: [planif.plan_fft_inverse(n1), planif.plan_fft_inverse(n2)],
            echelle: T::one() / T::from_usize(n1 * n2).unwrap().sqrt(),
        }
    }

    /// FFT 1D centrée en place : `ifftshift` (rotation à gauche de n / 2), FFT, `fftshift` (rotation à droite).
    fn centree(fft: &dyn Fft<T>, v: &mut [Complex<T>]) {
        let h = v.len() / 2;
        v.rotate_left(h);
        fft.process(v);
        v.rotate_right(h);
    }

    /// Applique la transformée (directe ou inverse) à `coupe`, en place.
    fn appliquer(&self, coupe: &mut [Complex<T>], inverse: bool) {
        let plans = if inverse { &self.inverse } else { &self.directe };
        for ligne in coupe.chunks_exact_mut(self.n2) {
            Self::centree(plans[1].as_ref(), ligne);
        }
        let mut colonne = vec![Complex::new(T::zero(), T::zero()); self.n1];
        for j in 0..self.n2 {
            for (a, c) in colonne.iter_mut().enumerate() {
                *c = coupe[a * self.n2 + j];
            }
            Self::centree(plans[0].as_ref(), &mut colonne);
            for (a, c) in colonne.iter().enumerate() {
                coupe[a * self.n2 + j] = *c * self.echelle;
            }
        }
    }
}

/// Modèle direct et adjoint pour une grille `[N₀, N₁, N₂]` (lecture le long de l'axe 0) et des temps de lecture donnés.
pub struct ModeleCpr<T: Reel> {
    dims: [usize; 3],
    /// Temps tᵢ (s) de chaque échantillon de lecture, longueur N₀.
    temps: Vec<T>,
    /// Lignes de la FFT 1D centrée normalisée le long de la lecture : `ligne[i * N₀ + x₀]` = cᵢ(x₀).
    ligne: Vec<Complex<T>>,
    fft2: Fft2Centree<T>,
}

impl<T: Reel> ModeleCpr<T> {
    /// Prépare le modèle pour une grille `dims` = `[N₀, N₁, N₂]` et les temps `temps` (s) des N₀ échantillons de lecture.
    ///
    /// Panique si `temps` n'a pas N₀ éléments.
    pub fn new(dims: [usize; 3], temps: &[T]) -> Self {
        let n0 = dims[0];
        assert_eq!(
            temps.len(),
            n0,
            "{} temps pour {n0} échantillons de lecture",
            temps.len()
        );
        let h = (n0 / 2) as i64;
        let norme = T::one() / T::from_usize(n0).unwrap().sqrt();
        let mut ligne = Vec::with_capacity(n0 * n0);
        for i in 0..n0 as i64 {
            for x0 in 0..n0 as i64 {
                // (i − h)(x₀ − h) modulo N₀ en entier : la phase reste exacte pour les grandes grilles
                let m = ((i - h) * (x0 - h)).rem_euclid(n0 as i64) as usize;
                let angle = -T::TAU() * T::from_usize(m).unwrap() / T::from_usize(n0).unwrap();
                let (s, c) = angle.sin_cos();
                ligne.push(Complex::new(c * norme, s * norme));
            }
        }
        ModeleCpr {
            dims,
            temps: temps.to_vec(),
            ligne,
            fft2: Fft2Centree::new(dims[1], dims[2]),
        }
    }

    fn verifier(&self, nom: &str, forme: &[usize]) {
        assert_eq!(forme, self.dims, "{nom} : forme {forme:?}, grille {:?}", self.dims);
    }

    /// Modèle direct `A(b0) u` : image `u` → k-space, `b0` en Hz, tous deux sur la grille `[N₀, N₁, N₂]`.
    ///
    /// Panique si une forme diffère de celle de la grille.
    pub fn direct(&self, u: &Array3<Complex<T>>, b0: &Array3<T>) -> Array3<Complex<T>> {
        self.verifier("image", u.shape());
        self.verifier("b0", b0.shape());
        let [n0, n1, n2] = self.dims;
        let (u, b0) = (u.as_standard_layout(), b0.as_standard_layout());
        let (u, b0) = (u.as_slice().unwrap(), b0.as_slice().unwrap());
        let coupe = n1 * n2;
        let coupes: Vec<Vec<Complex<T>>> = (0..n0)
            .into_par_iter()
            .map(|i| {
                let phase = -T::TAU() * self.temps[i];
                let mut s = vec![Complex::new(T::zero(), T::zero()); coupe];
                for x0 in 0..n0 {
                    let c = self.ligne[i * n0 + x0];
                    let debut = x0 * coupe;
                    for ((sj, uj), bj) in s
                        .iter_mut()
                        .zip(&u[debut..debut + coupe])
                        .zip(&b0[debut..debut + coupe])
                    {
                        let (sin, cos) = (phase * *bj).sin_cos();
                        *sj = *sj + c * *uj * Complex::new(cos, sin);
                    }
                }
                self.fft2.appliquer(&mut s, false);
                s
            })
            .collect();
        Array3::from_shape_vec((n0, n1, n2), coupes.concat()).unwrap()
    }

    /// Adjoint `Aᴴ(b0) k` : k-space `k` → image (reconstruction par phase conjuguée), `b0` en Hz.
    ///
    /// Panique si une forme diffère de celle de la grille.
    pub fn adjoint(&self, k: &Array3<Complex<T>>, b0: &Array3<T>) -> Array3<Complex<T>> {
        self.verifier("k-space", k.shape());
        self.verifier("b0", b0.shape());
        let [n0, n1, n2] = self.dims;
        let b0 = b0.as_standard_layout();
        let b0 = b0.as_slice().unwrap();
        let coupe = n1 * n2;
        // F₁₂⁻¹ de chaque ligne de lecture du k-space, indépendantes
        let lignes: Vec<Vec<Complex<T>>> = (0..n0)
            .into_par_iter()
            .map(|i| {
                let ki: ArrayView2<Complex<T>> = k.index_axis(Axis(0), i);
                let mut p: Vec<Complex<T>> = ki.iter().copied().collect();
                self.fft2.appliquer(&mut p, true);
                p
            })
            .collect();
        // chaque coupe x₀ de l'image somme les N₀ lignes dans l'ordre de i : résultat indépendant du nombre de fils
        let coupes: Vec<Vec<Complex<T>>> = (0..n0)
            .into_par_iter()
            .map(|x0| {
                let debut = x0 * coupe;
                let b = &b0[debut..debut + coupe];
                let mut s = vec![Complex::new(T::zero(), T::zero()); coupe];
                for (i, p) in lignes.iter().enumerate() {
                    let c = self.ligne[i * n0 + x0].conj();
                    let phase = T::TAU() * self.temps[i];
                    for ((sj, pj), bj) in s.iter_mut().zip(p).zip(b) {
                        let (sin, cos) = (phase * *bj).sin_cos();
                        *sj = *sj + c * *pj * Complex::new(cos, sin);
                    }
                }
                s
            })
            .collect();
        Array3::from_shape_vec((n0, n1, n2), coupes.concat()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Volume complexe pseudo-aléatoire déterministe dans [−1, 1]².
    fn aleatoire<T: Reel>(dims: [usize; 3], graine: u64) -> Array3<Complex<T>> {
        let mut etat = graine;
        let mut suivant = move || {
            etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            T::from_f64(((etat >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0).unwrap()
        };
        Array3::from_shape_simple_fn(dims, || Complex::new(suivant(), suivant()))
    }

    fn produit_scalaire<T: Reel>(a: &Array3<Complex<T>>, b: &Array3<Complex<T>>) -> Complex<f64> {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let p = x.conj() * y;
                Complex::new(p.re.to_f64().unwrap(), p.im.to_f64().unwrap())
            })
            .sum()
    }

    /// Critère 1 de P4 : ⟨A x, y⟩ = ⟨x, Aᴴ y⟩, grilles paires et impaires, champ de quelques centaines de Hz.
    fn adjoint_exact<T: Reel>(tolerance: f64) {
        for dims in [[7, 6, 5], [8, 5, 4]] {
            let temps: Vec<T> = (0..dims[0])
                .map(|i| T::from_f64(-2.5e-3 + 4.4e-5 * i as f64).unwrap())
                .collect();
            let modele = ModeleCpr::new(dims, &temps);
            let b0 = aleatoire::<T>(dims, 3).mapv(|z| z.re * T::from_f64(600.0).unwrap());
            let (x, y) = (aleatoire::<T>(dims, 1), aleatoire::<T>(dims, 2));
            let gauche = produit_scalaire(&modele.direct(&x, &b0), &y);
            let droite = produit_scalaire(&x, &modele.adjoint(&y, &b0));
            let ecart = (gauche - droite).norm() / gauche.norm();
            assert!(
                ecart < tolerance,
                "{dims:?} : écart {ecart:e} (⟨Ax, y⟩ = {gauche}, ⟨x, Aᴴy⟩ = {droite})"
            );
        }
    }

    #[test]
    fn l_adjoint_est_exact_en_f64() {
        adjoint_exact::<f64>(1e-12);
    }

    #[test]
    fn l_adjoint_est_exact_en_f32() {
        adjoint_exact::<f32>(1e-5);
    }

    /// Sans champ, A est la FFT 3D centrée normalisée, qui est unitaire : ‖A x‖ = ‖x‖ et Aᴴ A x = x.
    #[test]
    fn sans_champ_le_modele_est_une_fft_unitaire() {
        let dims = [7, 6, 5];
        let modele = ModeleCpr::new(dims, &[0.0f64; 7]);
        let b0 = Array3::<f64>::zeros(dims);
        let x = aleatoire::<f64>(dims, 4);
        let ax = modele.direct(&x, &b0);
        let norme = |a: &Array3<Complex<f64>>| a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!((norme(&ax) - norme(&x)).abs() / norme(&x) < 1e-12);
        let retour = modele.adjoint(&ax, &b0);
        let ecart = retour
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / norme(&x);
        assert!(ecart < 1e-12, "AᴴA x ≠ x : {ecart:e}");
    }
}
