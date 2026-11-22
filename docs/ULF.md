# Reconstruction IRM ultra-bas champ (48 mT) : portage en Rust

Objectif : porter dans medoxide la reconstruction des acquisitions TSE 3D de l'IRM tête ultra-bas champ (aimant
permanent, environ 48 mT), aujourd'hui en Python dans le dépôt de l'équipe `fridge-magnet-imaging` (fmi). Deux étapes :
(1) par paire d'acquisitions à lectures opposées (reg / inv), estimation du champ B0 et de l'image ; (2)
super-résolution à partir de trois plans. Plan d'ensemble, décisions et références Python : dépôt privé `oxide`,
`claude/pipeline-ulf.md`. Comme pour les autres modules, chaque étape se valide contre la référence Python par un
critère chiffré **écrit avant le code**.

## État des références (2026-11-20)

- Référence Python : fmi `feat/seg-sr` @ `cb7174b` ; sorties de l'étape 1 (méthode jointe B0 + image, réglage
  finest32, fantôme sub000 et subB) sur Garage (`spinloop/references/`), dont des opérateurs de test en
  complex128.
- Précision mesurée : en complex64, le modèle direct et son adjoint s'écartent de 2,3–2,8e-7 du complex128 ; l'étape 1
  peut tourner en `f32` (wgpu, Metal).
- Essais hors medoxide (`oxide/bench/ulf-essais.md`) : FFT 3D de toute taille sur GPU par Bluestein au-dessus de
  `burn-signal` (~6e-7 en f32) ; lecture ISMRMRD par `hdf5-metno` en statique, identique à h5py.

## P3 : lecture des acquisitions (`medoxide-ismrmrd`)

### Ce qu'il faut lire

Deux formats, tous deux avec un en-tête XML ISMRMRD :

| | Spinloop (`spinloop_recon.mrd.write`) | fmi (`fmi.io.writers.write_hdf5_file`) |
|---|---|---|
| fichier | ISMRMRD 1.x : `dataset/xml` (chaîne), `dataset/data` (tableau composé : `head`, `traj`, `data`) | `xml_header` (chaîne), `kspace` (tableau dense complex64) |
| échantillons | une acquisition par écho ; `idx.kspace_encode_step_1/2` décalés du centre ; lignes répétées numérotées par `idx.average` ; trains d'équilibre marqués `ACQ_IS_DUMMYSCAN_DATA` | `[lecture, phase 1, phase 2]` ; lignes non acquises (Fourier partiel) à zéro |
| polarité de lecture | paramètre `read_sign` (±1) ; échantillons tels qu'acquis | paramètre `invert_ro_polarity` ; **le k-space est retourné le long de la lecture à l'écriture** quand il vaut `True` |
| axes de la machine | paramètre `axes` (ex. `yzx` : lecture, phase 1, phase 2) | paramètre `plane` (ex. `yz`), convention de fmi |
| temps d'échantillonnage | paramètre `dwell_us` (µs) | paramètre `dwellTime` (ms) |
| porteuse | `H1resonanceFrequency_Hz` (porteuse jouée) | `H1resonanceFrequency_Hz` |

### Sortie commune

Un k-space 3D `[lecture, phase 1, phase 2]` en `Complex<f32>` (même ordre que fmi) ; le nombre de lectures par ligne
`[phase 1, phase 2]` (0 = non acquise ; Spinloop : lignes répétées **moyennées**, trains d'équilibre ignorés, comme
`spinloop_recon.cartesian.grid`) ; matrice, champ de vue (mm), centre du k-space, temps d'échantillonnage (µs),
porteuse (Hz), polarité de lecture, et tous les paramètres utilisateur (chaînes) avec l'en-tête XML brut pour la
provenance. Les axes (`axes`, `plane`) sont rendus tels quels : la convention commune des repères n'est pas encore fixée
(ADR 0029 de Spinloop, proposée) et ce lecteur ne l'invente pas.

### Critères (écrits avant le code)

1. **Spinloop** : le k-space lu est égal à celui de `spinloop_recon.cartesian.grid` (même transposition) à 1e-6
   relatif près (moyennes en `f32` contre `f64`), et le nombre de lectures par ligne est identique ; matrice, champ de
   vue, centre, `dwell_us`, porteuse, `read_sign` et `axes` égaux à ceux lus par `spinloop_recon.mrd.read`.
2. **fmi** : le k-space est égal **au bit près** à `h5py` (`f["kspace"][()]`) ; une ligne est acquise si l'un de ses
   échantillons est non nul (l'écrivain met à zéro les lignes non acquises ; le compte doit valoir celui de
   `np.any(k != 0, axis=0)`) ; `plane`, `invert_ro_polarity`, `echoShift` et TR égaux à ceux de
   `mo_data.header_params`.
3. **Tests sans données réelles** (dépôt public) : de petits fichiers synthétiques écrits par les deux écrivains Python,
   versionnés avec les valeurs attendues ; un test sur données réelles activé par `MEDOXIDE_ULF_DONNEES` (dossier de
   fichiers et empreintes calculées en Python), sur le modèle de `MEDOXIDE_EXIGER_DONNEES`.
4. **Plateformes** : compilation et tests sur macOS arm64, Linux aarch64 (PGX) et Linux x86_64 (RTX).

### Dépendances nouvelles (décision structurante, validée le 2026-11-20 avec le lancement de P3)

- `hdf5-metno` 0.15, features `static` (HDF5 compilé depuis les sources par cmake : aucune bibliothèque système, mais
  `cmake` requis à la compilation) et `complex` (lecture directe des complexes de h5py en `num_complex::Complex32`).
- `roxmltree` : lecture de l'en-tête XML (petit, sans dépendance). Rien dans le graphe actuel ne lit du XML (`xml-rs`
  n'y apparaît que pour la compilation sous Windows).

### Résultat (2026-11-20)

- Critères 1 et 2 : tenus sur les fichiers synthétiques (`tests/fixtures.rs`) ; deux mutations (train d'équilibre non
  ignoré, somme au lieu de la moyenne) font échouer le test Spinloop.
- Critère 3 : sur 17 fichiers réels (11 de fmi, dont 5 à lecture inversée ; 6 de Spinloop, dont 4 TSE 3D avec trains
  d'équilibre et lignes répétées), le k-space est **identique au bit près** au code Python (empreinte SHA-256), comme le
  compte des lignes, le temps d'échantillonnage, la porteuse, les axes ou le plan, la polarité et le TR. Aucun fichier Spinloop réel à lecture inversée (`read_sign` = -1) n'existe encore : ce cas n'est vérifié que sur le fichier synthétique. Le critère 1
  demandait 1e-6 : l'égalité exacte tient parce que la moyenne est faite comme en Python (somme en `f64`, division,
  conversion en `f32`).

## P4 : étape 1, modèle direct et adjoint (`medoxide-ulf`)

### Ce qu'on porte

`fmi.reconstruction.operators.cpr` (référence `cb7174b`), lecture le long de l'axe 0, `t_shift` = 0 (seuls cas utilisés
par la méthode jointe) :
- **modèle direct** `A(b0)` (image → k-space ; `utils_cpr.predict_kspace`, `time_segmented_cpr_adjoint` dans fmi) : pour
  chaque échantillon de lecture i, `k[i, :, :] = F(u · exp(−2iπ b0 tᵢ))[i, :, :]` ;
- **adjoint** `Aᴴ(b0)` (k-space → image, la CPR ; `utils_cpr.correct_image`, `time_segmented_cpr`) :
  `u = Σᵢ F⁻¹(Mᵢ k) · exp(+2iπ b0 tᵢ)`, Mᵢ ne gardant que la ligne i ;

où F est la FFT 3D **centrée** (`ifftshift`, `fftn`, `fftshift`) et **normalisée** (÷ √N ; F⁻¹ : × √N après `ifftn`),
et tᵢ le temps de l'échantillon i (constant le long des axes de phase). Inversion de nom à retenir : la fonction que fmi
appelle « adjoint » est le modèle direct de la méthode jointe.

### Critères (écrits avant le code)

1. **Test du produit scalaire** `⟨A x, y⟩ = ⟨x, Aᴴ y⟩` sur des volumes aléatoires : écart relatif ≤ 1e-12 en `f64`,
   ≤ 1e-5 en `f32`.
2. **Parité sur un petit cas synthétique versionné** (9 × 6 × 5, écrit par le code Python de fmi en complex128) :
   écart relatif L2 de `A x` et de `Aᴴ k` ≤ 1e-12 en `f64`, ≤ 1e-5 en `f32`.
3. **Parité sur les six plans de P1** (fixtures complex128 de `garage:spinloop/references/…`, test activé
   par `MEDOXIDE_ULF_P1`, données hors dépôt) : `y = A x` et `z = Aᴴ k_reg` à
   ≤ 1e-10 en `f64` (la référence a tourné sur GPU avec cuFFT) et ≤ 1e-5 en `f32` (P1 mesure 2,3–2,8e-7 entre
   complex64 et complex128 en PyTorch).
4. Durée d'une application sur 115 × 115 × 45 rapportée, sans critère (la référence Python tourne sur GPU).

### Résultat (2026-11-20)

- Critère 1 : produit scalaire exact à 1e-12 (`f64`) et 1e-5 (`f32`) sur des grilles paire et impaire ; sans champ, le
  modèle est la FFT unitaire (‖A x‖ = ‖x‖, Aᴴ A x = x à 1e-12).
- Critère 2 (cas synthétique 9 × 6 × 5 écrit par fmi) : tenu en `f64` et en `f32`.
- Critère 3 (six plans de P1, 115 × 115 × 45) : `A x` à 7,4–7,7e-16 et `Aᴴ k` à 6,1–6,3e-16 en `f64` (précision de la
  machine) ; 3,3–3,6e-7 et 2,9–3,0e-7 en `f32`.
- Durée d'une application sur le CPU du Mac (12 fils) : 0,05 s en `f64`, 0,03–0,04 s en `f32`, grâce à la
  factorisation (une FFT 2D par échantillon de lecture au lieu d'une FFT 3D).
- Trois mutations (signe de la ligne de FFT le long de la lecture, sens du décalage centré, signe de la phase du champ)
  font échouer les tests. La première passe le produit scalaire et l'unitarité : seule la parité avec fmi la voit.

### Suite (P4, estimation)

L'étape suivante est l'estimation jointe (B0 et image). Avant de la porter, le plan demande de comparer une base lisse
du champ (B-splines ou harmoniques sphériques) à l'INR de la référence, côté Python : c'est un choix de méthode, à
décider avant d'écrire le code.

### Choix de la représentation de B0 et de la précision (2026-11-21 / 22)

Étude Python (fmi, branche `p4/b0-base-lisse`, script `b0_bases_lisses.py` : critères et règle écrits avant les calculs,
amendements déclarés), sur les six plans de P1, RTX 2080 Ti :

- **Base lisse ou INR : l'INR reste.** Harmoniques sphériques (degré 4, 8) : très en dessous (LNCC 0,66 contre 0,80
  sur le premier plan), abandonnées. B-splines cubiques (nœuds tous les 40, 20, 10 mm) :
  - accord des champs entre plans **meilleur** que l'INR (corrélation +0,008, RMS −12 Hz ; dix fois le bruit des
    graines) ;
  - écart aux données de l'étape 1 **plus grand** de 7 à 33 % (au-delà du bruit, et à 400 itérations aussi : c'est la
    représentation, pas la convergence) ;
  - super-résolution du fantôme contre l'IRM 1,5 T : égale (erreur 10,46 % pour l'INR, 10,51 % pour B-splines 20 mm) ;
  - prédiction d'un plan laissé hors de la super-résolution : un peu moins bonne sur le fantôme (+0,4 à +1,6 %),
    nettement sur subB (+2 à +7 %, 10 mm à peine mieux que 20 mm) : le champ y varie vite près des bords.
  La règle et la prédiction du plan laissé dehors désignent l'INR (finest32, grille de hachage).
- **Précision : l'étape 1 peut tourner en `f32`.** La méthode jointe entière en float32 / complex64 donne un B0 à
  0,2–3,0 Hz du float64 (RMS dans l'objet), toujours sous l'écart entre deux graines (2,4–7,4 Hz), et une LNCC dans
  l'intervalle des graines ; l'écart aux données y est aussi, sauf sub000 xy (1,289 contre 1,287 au plus : 0,15 %).
  2,6 fois plus rapide (208 s contre 544 s par plan).
- **Graines** : `INRField` et `INRImage` remettent la graine à 0 dans leur constructeur ; l'étude passe la graine
  explicitement. Bruit des graines : LNCC ± 0,001, écart aux données ≤ 3 %, B0 2–9 Hz RMS.

Conséquence pour le portage : l'estimation jointe se porte avec une INR à grille de hachage, en `f32`, sur GPU.
Critère de parité de l'estimation (à écrire avant le code) : B0 et image finaux dans le bruit des graines.
