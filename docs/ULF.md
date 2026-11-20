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
