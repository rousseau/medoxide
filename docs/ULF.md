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

## P4 bis : estimation jointe B0 + image, conception (2026-11-22)

### Ce qu'on porte

`compare_methods.estimate_joint` (référence `cb7174b`) dans le réglage de P1 (`finest32`, surcharges de
`b0_sensitivity.overrides`), précision `f32` :

- **B0** : `INRField`. Grille de hachage de 8 niveaux, 2 caractéristiques par niveau, table de 2¹⁹, résolution 2 → 32
  (facteur (32/2)^(1/7)), puis un MLP de 64 neurones × 2 couches cachées (ReLU), et une sortie. Ensuite
  `forme = tanh(net)`, centrée et ramenée à une moyenne quadratique de 1 (dénominateur détaché du gradient),
  et `B0 = offset + amplitude · forme`, avec :
  - `amplitude = b0_min · (b0_max/b0_min)^sigmoid(s)` ;
  - `offset = b0_max · tanh(o)`.

  Les bornes viennent de `b0_bounds_hz` :
  - `b0_min` = 1/(2π T_lecture) ;
  - `b0_max` = 1/(2·dwell).
- **Image** : `INRImage`. Grille de 16 niveaux, résolution de base 16, facteur 2, table de 2¹⁹, MLP 64 × 2, deux
  sorties (parties réelle et imaginaire). Les données sont divisées par `data_scale`, le maximum des |F⁻¹ k| de reg
  et inv ; l'image est remultipliée par ce facteur à la fin.
- **Coordonnées** : `linspace(0, 1, n)` par axe (indexation `ij`).
- **Pertes** (`joint_losses`, mode `magnitude`, terme image `lncc`, `image_term_updates_b0 = False`, terme B0 `lncc`).
  La perte totale vaut :
  - 1 · fidélité, avec fidélité = ‖|F⁻¹k_reg| − |F⁻¹A(+b0)u|‖² + même terme pour inv avec −b0, en moyennes ;
  - \+ 1e-2 · image, avec image = −[LNCC(|u|, |Aᴴ(+b0)k_reg|) + LNCC(|u|, |Aᴴ(−b0)k_inv|)], les images corrigées
    étant détachées ;
  - \+ 1e-2 · b0, avec b0 = −LNCC(|Aᴴ(+b0)k_reg|, |Aᴴ(−b0)k_inv|) ;
  - \+ 1e-4 · lissage, avec lissage = Σ_axes moyenne((∂²B0 / h²)²) (`bending_penalty`, différences finies) ;
  - \+ 1 · support, avec support = moyenne(|u|²) hors du masque `object_support_mask` ;
  - \+ 1e-5 · (s² + o²).

  La LNCC utilise une fenêtre gaussienne de 4 mm (σ en voxels par axe). Le flou est séparable et à bords nuls.
  Le plancher vaut `eps · sqrt(var(a) · var(b))`, avec eps = 1e-6 et une variance **non biaisée** (défaut de
  `torch.var`) ; le gradient passe aussi par ce plancher.
- **Démarrage à chaud** (`prefit`) : 300 itérations d'Adam (lr 1e-2, constant), en moindres carrés relatifs :
  - le champ sur `b0_init` ;
  - l'image sur la moyenne des magnitudes corrigées (`averaged_corrected`, phase nulle).
- **Optimisation** : 200 itérations d'Adam (β = 0,9 / 0,999, **ε = 1e-8**, pas de décroissance des poids) :
  - lr 1e-3 pour les deux réseaux (tables et MLP) ;
  - lr 1e-2 pour s et o ;
  - décroissance cosinus `lr · (r + (1 − r) · ½ (1 + cos(π it / 200)))`, avec r = 1e-2.

Entrées : `k_reg`, `k_inv` (fenêtrées, sens de lecture rétabli), `t`, l'espacement, les bornes et `b0_init`.
`b0_init` est le champ de la méthode INR seule (« méthode B » : même champ, perte `lncc` seule, 400 itérations).
Pour la parité, ces entrées sont lues dans les `inputs.npz` de P1. La chaîne qui va du fichier ISMRMRD (P3) à ces
entrées (fenêtre de Tukey autour de l'écho, normalisation) et la méthode B viennent après ; elles réutilisent les
mêmes briques.

### Conception (validée le 2026-11-22 : opérateur sur CPU à gradient écrit ; `b0_init` lu dans P1, méthode B ensuite)

Crate `medoxide-ulf`, tenseurs de Burn 0.22 (type de device choisi à l'exécution, type flottant donné à chaque
création de tenseur). Les tests tournent sur le backend CPU `flex`, en `f64`
et en `f32` ; les calculs tournent sur wgpu (Metal) ou CUDA, en `f32`. Quatre modules :

1. **`inr`** : grille de hachage + MLP.
   - Les points de la grille sont fixes (les voxels). Les indices de table et les poids trilinéaires de chaque
     niveau se calculent donc **une fois, sur CPU, en `f64` et en entiers 64 bits**, avec la même arithmétique
     que `inr._fast_hash` :
     - produits masqués sur 32 bits, puis XOR, puis modulo la taille de table ;
     - niveaux denses indexés par pas ;
     - même modulo au bord.
   - L'encodage devient un `select` sur la table, pondéré, que l'autodiff de Burn sait dériver (accumulation
     dans la table).
   - Mémoire sur 115 × 115 × 45 : environ 0,6 Go pour l'image (16 niveaux × 8 coins × un indice et un poids) et
     0,3 Go pour le champ.
   - Calculer les coordonnées en `f64` évite la perte de précision du `f32` aux niveaux fins : l'échelle du
     niveau 15 de l'image vaut 524 287, et un pas de voxel en `f32` y perd environ 3 % de la fraction.
   - Initialisation propre à Burn :
     - tables uniformes ± 1e-4 ;
     - couches linéaires à ± 1/√fan_in, comme `nn.Linear` (à vérifier dans Burn).
   - L'écart à PyTorch est alors du bruit de graine, déjà mesuré.
2. **`cpr_diff`** : le modèle validé en P4 (CPU, `rustfft`) devient une opération à gradient propre (trait
   `Backward` de `burn-autodiff`).
   - Les complexes sont des paires réelles côté Burn : aucune convention de gradient complexe à reproduire.
   - Les dérivées par rapport au champ n'ont besoin que d'un opérateur de plus, l'adjoint pondéré par le temps
     `Aₜᴴ k = Σᵢ tᵢ F⁻¹(Mᵢ k) exp(+2iπ b0 tᵢ)`, calculé dans la même passe que `Aᴴ` :
     - pour y = A(b0) u et un gradient g sur y :
       - ∂/∂u = Aᴴ g ;
       - ∂/∂b0 = Re(−2iπ · u · conj(Aₜᴴ g)) ;
     - pour z = Aᴴ(b0) k et un gradient g sur z :
       - ∂/∂b0 = Re(conj(g) · 2iπ · Aₜᴴ k) ;
       - ∂/∂k = A g (non utilisé : les données sont fixes).
   - Rien n'est gardé par segment de temps : la mémoire ne dépend pas du nombre d'échantillons de lecture. C'est
     l'équivalent du `checkpoint` qu'il a fallu ajouter en Python pour tenir sur la RTX.
   - Coût par itération : environ 6 passes (2 `A`, 2 `Aᴴ` + `Aₜᴴ`, 2 passes au retour), soit 0,2 à 0,3 s sur CPU
     au rythme mesuré en P4 (0,03–0,05 s par passe), contre environ 1 s par itération pour la référence Python `f32` sur
     GPU (208 s par plan, démarrage à chaud compris). Les transferts CPU ↔ GPU portent sur 5 Mo par volume.
3. **`pertes`** :
   - LNCC avec flou séparable écrit comme un produit par une matrice de Toeplitz le long de chaque axe (bords nuls
     exacts, dérivable sans noyau dédié) ;
   - fidélité, lissage, support, pénalité des scalaires ;
   - masque du support (flou, seuil, dilatation par `max_pool3d`) et `data_scale`, calculés une fois, hors gradient.
4. **`jointe`** :
   - le démarrage à chaud puis la boucle ;
   - deux optimiseurs Adam de Burn (réseaux ; scalaires), puisqu'il y a un lr par groupe ; Adam étant par
     paramètre, c'est équivalent à un seul Adam à deux groupes ;
   - ε fixé à 1e-8 (Burn a 1e-5 par défaut, ce qui écraserait les gradients des tables, initialisées à 1e-4) ;
   - la décroissance cosinus est appliquée à la main.

Autre voie envisagée, non retenue pour la première version : tout sur GPU, opérateur compris (FFT de Bluestein
au-dessus de `burn-signal`, 0,2–0,6 s par FFT 3D en 230³ d'après les essais), gradient par l'autodiff de Burn. Elle
garderait en mémoire les intermédiaires de chaque segment (c'est ce qui dépassait les 11 Go en Python) et serait
plus lente que `rustfft` sur CPU aux tailles utiles. On y reviendra si le profil de la première version le
justifie.

### Critères (écrits avant le code)

Les cas de test sont écrits par un script Python (`scripts/cas_jointe.py`, fmi `cb7174b`) :
- un petit cas synthétique versionné (grille 9 × 6 × 5, champ et image INR initialisés avec la graine 0 ; aucune
  donnée de sujet) ;
- les six plans de P1 (activés par `MEDOXIDE_ULF_P1`, hors dépôt).

1. **INR** (mêmes poids, chargés depuis le script) :
   - indices de table identiques à ceux de PyTorch, entier pour entier ;
   - sorties du champ et de l'image à ≤ 1e-12 en `f64` et ≤ 1e-6 en `f32` (écart relatif L2) ;
   - gradient d'une fonctionnelle linéaire aléatoire par rapport à tous les paramètres à ≤ 1e-10 / 1e-5.
2. **Opération dérivable** :
   - gradients par rapport à u et b0 contre des différences finies (`f64`, ≤ 1e-6) et contre l'autograd de PyTorch,
     sur le cas synthétique (≤ 1e-10 en `f64`, ≤ 1e-5 en `f32`) et sur les plans de P1 (≤ 1e-8 / 1e-4) ;
   - une mutation (signe du facteur 2iπ dans le gradient en b0) doit faire échouer ce test.
3. **Pertes** : chaque terme et son gradient par rapport à (u, b0), évalués sur les sorties de P1 :
   - valeurs à ≤ 1e-10 (`f64`) et 1e-5 (`f32`) ;
   - gradients à ≤ 1e-8 / 1e-4 ;
   - même masque de support, voxel pour voxel.
4. **Un pas d'optimisation** : partant des poids exportés de PyTorch, les paramètres après un pas de démarrage à
   chaud et après un pas de la boucle jointe sont à ≤ 1e-10 (`f64`) des valeurs de PyTorch. Au-delà, l'écart grandit
   sans signification (optimisation non convexe) ; c'est le critère 5 qui juge.
5. **Bout en bout, `f32` sur GPU, initialisation de Burn**, sur les six plans de P1 :
   - B0 à moins du plus grand écart entre graines de ce plan (RMS dans l'objet, de 2,4 à 7,4 Hz selon le plan) de la
     référence `f64` ;
   - LNCC et écart aux données dans l'enveloppe des graines 0, 1, 2 en `f64` et du calcul `f32` de PyTorch
     (l'enveloppe inclut ce dernier, puisqu'il sortait de 0,15 % de l'intervalle des graines sur sub000 xy).
6. Durée rapportée, sans critère, sur le Mac (Metal), la RTX (CUDA) et le CPU seul ; référence : 208 s par plan pour
   PyTorch `f32` sur la RTX.

Ordre : 1 → 2 → 3 → 4 → 5. Chaque critère est tenu avant de passer au suivant ; les mutations qui le font échouer
sont notées.

### Résultat du critère 1 (2026-11-22) : INR à grille de hachage

Module `inr` (`ConfigGrille`, `Encodage`, `InrHachage`, `ChampB0`, `ImageInr`). Cas de test écrits par
`scripts/cas_inr.py` (fmi `p4/b0-base-lisse`, `inr.py` et `utils_cpr.py` identiques à `cb7174b`) :
- cas synthétique 9 × 6 × 5, versionné, à petites tables, avec des niveaux denses et des niveaux hachés ;
- cas aux réglages de P1 sur 115 × 115 × 45 (1,4 Go, hors dépôt, sans donnée de sujet ; `MEDOXIDE_ULF_INR`).

Les tables sont tirées dans [−1, 1] pour que la sortie varie vraiment.

- **Indices de table** : identiques à ceux de PyTorch, entier pour entier, sur tous les niveaux des deux cas.
- **Coordonnées** : identiques bit pour bit. `torch.linspace` calcule chaque valeur en une multiplication-addition
  fusionnée (FMA, un seul arrondi), sur CPU arm64, CPU x86_64 et CUDA (vérifié pour n = 2 à 299). Sans FMA, 3 valeurs
  sur 45 diffèrent d'un ulp, et le test des coordonnées échoue sur le cas P1.
- **`f64` (CPU)** :
  - sorties à 3,6e-15 (synthétique) et 1,1e-15 (P1) ;
  - gradients à 3,3e-14 et 4,6e-12.
- **`f32`** :
  - synthétique : sorties à 8,1e-7 et gradients à 4,0e-6 sur CPU, sous les seuils initiaux (wgpu : 2,1e-6 et 4,4e-6) ;
  - P1, CPU / wgpu (Metal) :
    - B0 à 6,1e-7 / 9,9e-7 (PyTorch `f32` : 1,4e-6) ;
    - image à 5,5e-8 (PyTorch `f32` : 1,6e-3, ses coordonnées `f32` perdant la fraction aux niveaux fins) ;
    - gradients à 9,9e-3 / 1,0e-2 au plus, au niveau de PyTorch `f32` ou en dessous pour chaque paramètre.
- **Amendement du critère `f32`** (2026-11-22, déclaré après le premier dépassement, avant les corrections) : en `f32`,
  chaque sortie et chaque gradient doit être sous le seuil initial **ou** sous deux fois l'écart de PyTorch `f32` au
  float64 sur le même cas (mesuré par le script), le plus grand des deux. Les seuils initiaux (1e-6, 1e-5) sont
  intenables aux réglages de P1 pour deux raisons :
  - le gradient du biais final du champ est une compensation presque complète, parce que le centrage rend B0 presque
    insensible à un décalage uniforme : κ = Σ|termes| / |somme| = 9,5e5, soit κ · u ≈ 6e-2 en `f32` ;
  - PyTorch `f32` lui-même s'écarte de 1,1e-2 sur ce biais, et jusqu'à 11 % sur les gradients de l'image.
- **Trois corrections numériques en route**, chacune mesurée :
  - (a) **sommes à deux niveaux** : moyennes du champ (B0 en `f32` : 8,8e-5 → 6,1e-7) et diffusion des scalaires en deux
    étapes (gradient de l'offset : 1,0e-5 → 2,5e-7). Au retour d'une diffusion, Burn somme le gradient en une seule
    boucle par valeur diffusée, sur CPU comme sur GPU ;
  - (b) **MLP par blocs de 1 024 points** (produit matriciel par lots, points complétés par des points fictifs de poids
    nul) : les gradients des poids et des biais se somment aussi à deux niveaux (gradient des poids de la dernière
    couche du champ : 4,0e-5 → 1,6e-5, PyTorch 1,9e-5) ;
  - (c) **contournement d'un défaut de `burn-autodiff` 0.22.1** : au retour, `float_div_scalar` calcule l'inverse du
    diviseur en `f32`, même pour un tenseur `f64` (gradient limité à environ 1e-8 relatif). On multiplie donc par
    l'inverse.
- **Mutations** :
  - font échouer le test : permutation des nombres premiers du hachage, pas dense transposé, décalage + 0,5 retiré,
    ordre des coins inversé, dénominateur du champ non détaché, `linspace` sans FMA (cas P1) ;
  - équivalente : le masque 32 bits du hachage, puisqu'une table hachée a toujours une taille en puissance de 2 et
    que le modulo ne garde que les bits bas.
- **Mémoire** (cas P1) : l'encodage précalculé de l'image tient environ 0,6 Go en `f32` (indices et poids), comme
  prévu.
- **Plateformes** (cas synthétique, Linux x86_64, RTX) :
  - `f64` sur CPU : 4,6e-15 et 3,6e-14 ;
  - `f32` sur CPU : tenu ;
  - `f32` sur wgpu (Vulkan) et sur CUDA : **un échec strict**, le gradient du biais final du champ, à 1,40e-5 pour une
    limite de 1,20e-5 (2 × PyTorch `f32`, 6,0e-6). Sur ce même cas, le CPU de la RTX est à 1,19e-5 et celui du Mac à
    1,75e-6 : sur cette quantité mal conditionnée (κ ≈ 100 sur 270 points), l'écart `f32` est de l'arrondi qui varie
    d'un matériel à l'autre d'un facteur 8. Noté comme échec, sans second amendement.
- **Décision (2026-11-22)** : l'échec strict est accepté ; on passe au critère 2.

### Changement de conception (2026-11-22) : gradient de l'opérateur par coupure du graphe

La conception prévoyait une opération Burn à retour propre (trait `Backward` de `burn-autodiff`). En Burn 0.22, une
telle opération passe par une extension de backend générée par macro (`backend_extension`), à implémenter pour
chaque backend (`Flex`, `Cube`) et à nouveau pour l'autodiff. On fait plus simple, sans dépendre du backend : on
**coupe le graphe à la sortie des réseaux**.

1. Les sorties des réseaux (u, b0) sont recopiées en feuilles détachées. Les pertes partent de ces feuilles, et les
   sorties de l'opérateur, calculées sur CPU, sont elles aussi des feuilles.
2. Un premier `backward` donne les gradients des pertes par rapport à ces feuilles.
3. Les gradients écrits à la main (ci-dessous) ramènent les gradients des sorties de l'opérateur sur u et b0, et
   s'ajoutent aux gradients directs.
4. Un second `backward`, sur Σ (u, b0) · gradient détaché, les propage dans les réseaux.

C'est le même calcul que l'opération à retour propre. Seule la mise en œuvre change : deux `backward` par itération.
Elle sera validée au critère 3, avec les pertes.

### Résultat du critère 2 (2026-11-22) : gradients de l'opérateur

`ModeleCpr::adjoint_et_pondere` calcule `Aᴴ k` et `Aₜᴴ k` dans la même passe, et `adjoint` reste inchangé (parité
de P4 conservée). Deux fonctions donnent les gradients :
- `gradient_direct` : pour y = A(b0) u, ∂L/∂u = Aᴴ g et ∂L/∂b0 = 2π Im(u · conj(Aₜᴴ g)) ;
- `gradient_adjoint_b0` : pour z = Aᴴ(b0) k, ∂L/∂b0 = −2π Im(conj(g) · Aₜᴴ k).

On y note g = ∂L/∂Re + i ∂L/∂Im (la convention de PyTorch). Références écrites par `scripts/cas_cpr.py gradients`,
avec l'autograd de PyTorch sur les fonctions de fmi : pertes L_y = Σ Re(conj(w_y) · A(b0) x) et
L_z = Σ Re(conj(w_z) · Aᴴ(b0) k), à poids aléatoires.

| Cas | `f64` contre PyTorch | `f32` contre PyTorch | Différences finies (`f64`) |
|---|---|---|---|
| Synthétique 9 × 6 × 5 (versionné) | 8,3e-16 – 9,4e-16 | 4,8e-7 – 6,0e-7 | ≤ 8,4e-9 |
| Six plans de P1 (`MEDOXIDE_ULF_P1_GRAD`) | 7,3e-16 – 8,7e-16 | 3,0e-7 – 4,2e-7 | ≤ 1,7e-8 |

Tous les seuils sont tenus : 1e-10 / 1e-5 sur le synthétique, 1e-8 / 1e-4 sur P1, 1e-6 pour les différences finies.
Quatre mutations font échouer les tests du cas synthétique, PyTorch comme différences finies :
- signe du facteur 2iπ dans chacun des deux gradients ;
- pondération par le temps oubliée ;
- conjugué déplacé.

### Résultat du critère 3 (2026-11-22) : pertes et coupure du graphe

Module `pertes` :
- `Donnees` prépare une fois, par paire reg / inv, le facteur `echelle`, les k-space normalisés, les magnitudes
  observées |F⁻¹k|, le masque de support (calculé en `f64`) et les matrices du flou de la LNCC ;
- `evaluer` donne les cinq termes non pondérés et la coupure ;
- `Coupure::gradients` ramène sur u et b0 les gradients des feuilles placées à la sortie du modèle.

La LNCC passe par un flou séparable écrit en produits matriciels (bords nuls exacts), une variance non biaisée et les
sommes à deux niveaux du critère 1. F⁻¹ et F sont le modèle sans champ (`Aᴴ(0)`, `A(0)`). Références :
`scripts/cas_pertes.py`, avec l'autograd de PyTorch en float64 sur `joint_losses` réglé comme `estimate_joint`, plus
`bending_penalty` et la pénalité hors support :
- cas synthétique 32 × 24 × 16, versionné (objet simulé par le modèle direct, avec bruit) ;
- sorties de l'étape 1 de P1 sur les six plans (`MEDOXIDE_ULF_P1_PERTES`, hors dépôt).

- **Masque de support** : identique voxel pour voxel. **Facteur d'échelle** : à 1e-12.
- **`f64`** :
  - valeurs à 6,5e-15 au plus ;
  - gradients à 5,8e-13 au plus, à travers la coupure.
- **`f32`** :
  - valeurs à 1,1e-6 au plus, gradients de la fidélité, du terme B0, du lissage et du support sous 1e-4 ;
  - le gradient du terme image par rapport à u est à 8,1e-5 – 3,4e-4. PyTorch `f32` fait 9,5e-5 – 2,6e-4 sur le même
    calcul : il dépasse lui aussi le seuil de 1e-4 sur cinq plans sur six, et Rust reste à 1,4 fois PyTorch au plus.
    C'est le conditionnement de la LNCC : la variance locale y est calculée comme E[a²] − E[a]², sujette aux pertes de
    chiffres en `f32`, et la formule est celle de la référence.
- **Amendement du critère `f32`** (2026-11-22, déclaré après le premier dépassement, avant la mesure des autres plans) :
  même règle qu'au critère 1. En `f32`, chaque valeur et chaque gradient doit être sous le seuil initial ou sous deux
  fois l'écart de PyTorch `f32` au float64 sur le même cas (mesuré par le script). Tenu sur les six plans et sur le cas
  synthétique.
- **Mutations** :
  - font échouer le test : images corrigées non détachées dans le terme image, signe de la contribution de inv au
    gradient de b0, F⁻¹ au lieu de F au retour de la fidélité, dilatation du masque permutée, lissage sans division
    par h² ;
  - survit : variance biaisée dans le plancher de la LNCC. Le plancher (1e-12 · var · var) n'agit que si la variance
    locale tombe à ce niveau, et n − 1 au lieu de n le change de 1e-4 en relatif : la mutation est presque
    équivalente sur ces données.
- Le second `backward` de la coupure (propagation dans les réseaux) sera exercé au critère 4.

### Changement de conception : Adam écrit dans le module (2026-11-22)

L'Adam de Burn 0.22.1 garde β₁, β₂ et ε en `f32`, corrections de biais comprises : 1 − 0,9f32 diffère de 0,1 de
2,4e-7 en relatif, et le paramètre après un pas s'écarte d'autant de PyTorch, même en `f64`. Le module `jointe` reprend
donc `torch.optim.Adam` (`_single_tensor_adam` : `lerp` du premier moment, `addcmul` du second, ε ajouté à
√v / √(1 − β₂ᵗ)) avec des scalaires `f64`. Le lr est donné à chaque pas, ce qui couvre les groupes (réseaux à 1e-3,
scalaires du champ à 1e-2) et la décroissance cosinus, sans deux optimiseurs.

### Résultat du critère 4 (2026-11-22) : pas d'optimisation

Module `jointe` : `Adam`, `pas_prefit_champ`, `pas_prefit_image`, `pas_joint` (les deux `backward` de la coupure),
`estimer` (méthode complète, pas encore exercée de bout en bout). Références : `scripts/cas_jointe.py`, qui reprend
`estimate_joint` en float64 sur CPU avec les poids initiaux de fmi (graine 0) :
- 2 pas de démarrage à chaud pour le champ (cible b0_init), puis 2 pour l'image (cible `averaged_corrected`) ;
- depuis les mêmes poids initiaux, le démarrage complet (300 pas par réseau), puis 2 pas joints ;
- deux pas plutôt qu'un, pour exercer les corrections de biais d'Adam et la décroissance cosinus au-delà du premier.

Cas : synthétique 32 × 24 × 16, versionné (réseaux à petites tables, comme au critère 1) ; six plans de P1 aux
réglages de P1 (`MEDOXIDE_ULF_P1_JOINTE`, 600 Mo par plan, hors dépôt puisqu'ils contiennent les k-space des
sujets).

**Point de départ des pas joints.** Ils partaient d'abord de l'état après 2 pas de démarrage. Le test a échoué sur le
cas synthétique : terme image à 8,9e-9 de PyTorch avant tout pas, image à 5,3e-9 après le premier pas. Cause : à cet
état, |u| vaut 0,202 à 1,5e-5 près (écart-type relatif). La variance locale de la LNCC, E[a²] − E[a]², y perd une
dizaine de chiffres ; PyTorch lui-même bouge de 3 à 5e-10 quand |u| est recalculé par √(re² + im²) au lieu de `abs`.
Cet état n'arrive jamais dans la méthode, où la boucle jointe part toujours du démarrage complet. Les pas joints partent
donc des poids exportés par PyTorch après le démarrage complet, ce que permet la lettre du critère (« partant des
poids exportés de PyTorch ») ; le seuil est inchangé.

**Démarrage du champ, amendement du 2026-11-22** (déclaré après l'échec de `sub000_yz`, avant la mesure des trois
plans de subB). À l'initialisation de fmi, la sortie du réseau du champ est presque constante (−0,029 à 2,7e-6 près
sur `sub000_yz`) ; la forme du champ est cette sortie centrée puis ramenée à une moyenne quadratique de 1, et le
centrage perd des chiffres. PyTorch ne se reproduit pas lui-même à 1e-10 : entre ses calculs à un thread et à huit
threads (seul l'ordre des sommes change), les paramètres du champ après les deux pas diffèrent jusqu'à 6,1e-10 sur ce
plan. Ce point de départ est celui de la méthode et ne peut pas être déplacé. Règle, la même qu'aux critères 1 et 3 :
sur les deux démarrages, un paramètre passe s'il est sous 1e-10 ou sous deux fois cet écart de PyTorch à lui-même,
mesuré paramètre par paramètre par le script (`cas_jointe.py plancher`). Les pas joints gardent le seuil de 1e-10.

- **Cas synthétique** :
  - démarrage : champ à 8,2e-13, image à 1,3e-16 ; cible de l'image à 6,0e-16 ;
  - pas joints : termes de la perte à 1,6e-15 au plus, paramètres du champ et de l'image à 2,0e-15 au plus (5,3e-14
    rapporté au déplacement du pas).
- **Six plans de P1** :
  - cible du démarrage de l'image à 6,2e-16 au plus, image après son démarrage à 1,2e-14 au plus ;
  - pas joints : termes de la perte sous 1e-10 partout (de 1e-16 à 1,4e-12), paramètres à 2,1e-11 au plus (champ) et
    1,0e-13 au plus (image) ;
  - champ après son démarrage : 2,4e-11, 2,6e-11, 7,0e-11 et 3,2e-11 sur quatre plans, sous 1e-10 ; 3,4e-10 sur
    `subB_xy`, sous sa limite amendée (plancher de PyTorch 2,3e-10) ;
  - **manqué sur `sub000_yz`** : le champ est à 7,5e-10 (PyTorch contre lui-même : 6,1e-10 au plus), mais deux
    paramètres dépassent leur limite amendée, les biais des couches 0 et 1 du réseau du champ : 2,2e-10 et 2,1e-10,
    pour des planchers de 7,3e-11 et 3,3e-11 (limites 1,5e-10 et 1,0e-10). Le plancher est mesuré sur une seule
    perturbation (un thread contre huit) et sous-estime donc la dispersion paramètre par paramètre ; Rust somme dans
    un autre ordre encore (sommes à deux niveaux). Écart accepté par François le 2026-11-22 ; le test le tolère pour ces
    deux paramètres seulement, plafonné à 3e-10.
- **Mutations**, toutes détectées sur le cas synthétique :
  - ε = 1e-5 (défaut de Burn), β₁ et β₂ en `f32` (2,6e-8), sans correction de biais du premier moment ;
  - lr des scalaires égal à celui des réseaux, décroissance cosinus décalée d'un pas ;
  - pénalité des scalaires absente du second `backward`, gradient direct de b0 oublié dans la coupure, poids du
    lissage 1e-2 au lieu de 1e-4.
- Durées sur le Mac (CPU, `f64`) : 30 à 60 s par plan pour le test Rust ; 29 min par plan pour le script PyTorch, dont
  l'essentiel est le démarrage complet (600 pas).

### Résultat du critère 5 (2026-11-22) : bout en bout, `f32` sur GPU — non tenu tel qu'écrit, tenu amendé

Exemple `jointe_p1` (méthode complète : démarrage à chaud de 300 pas par réseau, 200 pas joints ; modèle CPR sur CPU en
`f32`, réseaux et pertes sur GPU en `f32`) ; comparaison par `scripts/compare_jointe_p1.py` aux sorties de
`b0_bases_lisses.py` (graines 0, 1, 2 en float64 et calcul float32 de PyTorch, sur la RTX). Les métriques sont recalculées
avec le code de l'étude ; contrôles : la LNCC du calcul float32 de PyTorch est retrouvée à 1e-16, et σ (tiré de ce
calcul) redonne l'écart aux données de la sortie de P1 à 4e-9.

**Correction préalable.** Les premiers essais duraient des heures : au retour, burn-cubecl 0.22.1 somme le gradient de
`select` avec un fil par colonne de la table, qui parcourt les 4,7 millions d'indices du niveau (38 s par passe pour
l'image sur Metal, 12 s par pas du champ sur CUDA). La lecture des tables passe désormais par `gather`, dont le retour
est une addition atomique par valeur : 0,12 s. Écarts des critères 1 et 4 inchangés.

**Critère tel qu'écrit (une exécution, initialisation de Burn, graine 0, CUDA)** : tenu sur deux plans sur six.

| plan | B0 à la graine 0 (seuil) | LNCC (enveloppe) | écart aux données (enveloppe) |
|---|---|---|---|
| sub000 xy | 3,56 Hz (2,71) ✗ | 0,8276 (0,8277–0,8281) ✗ | 1,2905 (1,2832–1,2893) ✗ |
| sub000 yz | 3,47 Hz (2,55) ✗ | 0,7946 (0,7947–0,7948) ✗ | 2,1357 ✓ |
| sub000 zx | 2,77 Hz (3,11) ✓ | 0,8137 ✓ | 1,5765 (1,5810–1,5876) ✗ |
| subB xy | 3,34 Hz (3,10) ✗ | 0,6253 ✓ | 2,6204 (2,6463–2,6604) ✗ |
| subB yz | 3,65 Hz (8,62) ✓ | 0,6498 ✓ | 2,3803 ✓ |
| subB zx | 3,91 Hz (6,08) ✓ | 0,6419 ✓ | 2,4237 ✓ |

Les sorties d'enveloppe vont dans les deux sens (sur sub000 zx et subB xy, Rust colle mieux aux données que les trois
graines). Ce que montrent les contrôles :
- **Mêmes poids initiaux que fmi** (graine 0, option `--poids`), CUDA : B0 à 1,2–3,4 Hz de la graine 0, sous le seuil
  sur les six plans ; LNCC hors enveloppe de 1e-4 au plus sur trois plans, écart aux données de 0,08 % sur un. Le
  portage reproduit la méthode : l'écart du tableau vient de l'initialisation, qui se comporte comme une autre graine.
  (B0 plus loin de la graine 0 que le calcul float32 de PyTorch sur certains plans : 2,07 Hz contre 0,17 sur sub000 zx ;
  800 pas d'Adam amplifient les différences d'arrondi et l'ordre non déterministe des additions atomiques.)
- **Graines de Burn 0, 1, 2** (CUDA, Metal, Metal) : écart de B0 entre deux graines de Rust 1,9–8,9 Hz, contre 1,8–8,6 Hz
  entre graines de PyTorch ; moyenne des trois B0 de Rust à 1,3–2,3 Hz de celle de PyTorch. LNCC moyenne à 1,7 écart-type
  au plus de celle de PyTorch ; écart aux données moyen à moins d'un écart-type, sauf sub000 xy (+0,5 %, +3,2 écarts-types,
  une comparaison sur douze avec trois graines de chaque côté).
- Une quatrième graine sort de l'étendue de trois graines une fois sur deux : le critère, écrit sur une seule exécution
  contre l'enveloppe de trois graines, ne pouvait pas distinguer un portage fidèle du bruit d'initialisation.

**Amendement du critère 5** (décidé par François le 2026-11-22, après la mesure ; motif : le critère d'origine compare
une seule exécution à l'étendue de trois graines et ne distingue pas un portage fidèle du bruit d'initialisation). Le
critère devient :
1. avec les poids initiaux de fmi (graine 0), B0 à moins du plus grand écart entre graines de la référence `f64`, sur les
   six plans (fidélité du portage, à initialisation égale) ;
2. avec l'initialisation de Burn, sur trois graines : écart de B0 entre graines de Rust du même ordre que celui de
   PyTorch, et moyennes (B0, LNCC, écart aux données) compatibles avec celles des graines de PyTorch, compte tenu du
   bruit des graines (absence d'écart systématique).

**Tenu** avec les mesures ci-dessus : (1) B0 à 1,2–3,4 Hz pour des seuils de 2,5–8,6 Hz ; (2) écarts entre graines
1,9–8,9 Hz contre 1,8–8,6 Hz, moyennes de B0 à 1,3–2,3 Hz, LNCC et écart aux données à moins de 2 écarts-types sauf
l'écart aux données de sub000 xy (+0,5 %, 3,2 écarts-types sur trois graines), noté. Les LNCC hors enveloppe de 1e-4 au
plus restent notées.

### Résultat du critère 6 (2026-11-22) : durées

Plan sub000 xy (subB yz pour la seconde mesure CUDA), méthode complète en `f32` (`jointe_p1`, durées par phase dans
le fichier de sortie), dispositif libre de tout autre calcul. Les mesures du critère 5 sur la RTX ne comptent pas : un
reste d'un calcul arrêté (version à `select`) occupait le GPU pendant ce temps (`suivi --arrete` ferme la session sans
tuer le processus).

| dispositif | total | démarrage champ + image | boucle jointe (200 pas) | dont opérateur CPR* |
|---|---|---|---|---|
| RTX, CUDA | 240 s, 245 s | 16 + 26 s | 193 s, 198 s | 170 s, 175 s |
| Mac, Metal | 283 s | 49 + 70 s | 161 s | 103 s |
| Mac, CPU seul | 1 295 s | 229 + 467 s | 596 s | 70 s |
| RTX, CPU seul | 4 189 s | 871 + 1 539 s | 1 771 s | 182 s |
| référence : PyTorch `f32`, RTX | 208 s | | | |

\* Durée de `evaluer` et `Coupure::gradients` : modèle CPR sur CPU et transferts, mais aussi, sur GPU, l'attente des
calculs du réseau encore en file au moment du premier transfert ; c'est donc une borne haute de la part de l'opérateur
(sur CPU seul, où rien n'est en file : 70 s sur le Mac, 0,35 s par pas ; 182 s sur la RTX, 0,91 s par pas).

Lecture :
- Sur la RTX, Rust est 15 % plus lent que PyTorch ; la boucle jointe y fait 0,97 s par pas, dont 0,85 s comptés dans
  l'opérateur, qui coûte bien cela sur son CPU (0,91 s par pas en CPU seul) : le CPU x86_64 de la RTX fait les FFT
  2,6 fois plus lentement que celui du Mac, et l'estimation de la conception (0,2–0,3 s par pas) valait pour le Mac. Le démarrage à chaud (sans opérateur) est rapide (42 s). Le gain viendra de l'opérateur : FFT sur GPU
  (la voie écartée en conception) ou recouvrement du calcul CPU et GPU.
- Le Mac (Metal) est plus lent que la RTX sur les réseaux (119 s de démarrage contre 42 s), plus rapide sur l'opérateur.
- Sur CPU seul, les réseaux dominent (backend CPU de Burn) : facteur 5 sur le Mac, 17 sur la RTX, où il n'occupe
  qu'environ deux cœurs sur douze.

## P4 ter : opérateur CPR sur le dispositif (2026-11-22)

**Constat** (critère 6) : sur la RTX, le modèle CPR sur CPU fait 0,85–0,91 s des 0,97 s d'un pas joint. Mesure par
passe (`chrono_cpr`, P1, `f32`) : RTX 36–82 ms, Mac 14–41 ms, dix passes par pas. Les `sin_cos` n'en font qu'une petite
part (direct avec champ 49 ms, sans champ 36 ms sur la RTX) : le coût est la somme le long de la lecture, en N₀² · N₁N₂
(68 millions de produits complexes par passe), qu'une optimisation du CPU ne réduirait que d'un facteur 2 ou 3.

**Conception.** Un second opérateur, `OperateurCpr`, fait les mêmes calculs en tenseurs Burn sur le dispositif des
réseaux (complexes en paires réel / imaginaire) :
- somme le long de la lecture : phases exp(∓2iπ b0 tᵢ) et produits par cᵢ(x₀) en tenseurs [échantillons, N₀, N₁N₂],
  par paquets d'échantillons (mémoire bornée), réduits par `sum_dim` ;
- FFT 2D centrée des coupes écrite comme deux produits matriciels par les matrices de la DFT centrée normalisée
  (115 × 115 et 45 × 45 : quelques dizaines de millions d'opérations, négligeable) ;
- passes sans champ (F et F⁻¹ de la fidélité) : F₀ aussi en produit matriciel, sans phases ;
- gradients : les formules écrites du critère 2 (pas d'autodiff à travers l'opérateur, la coupure du graphe reste) ;
- `Donnees`, `evaluer` et `Coupure` passent sur cet opérateur ; u, b0 et les gradients restent sur le dispositif
  (plus de transferts). `ModeleCpr` (CPU) reste la référence validée contre fmi.

**Critères (écrits avant le code)** :
1. `OperateurCpr` contre `ModeleCpr` (direct, adjoint, adjoint pondéré, deux gradients ; avec et sans champ ; grilles
   paires et impaires et taille P1) : ≤ 1e-12 en `f64` (CPU), ≤ 1e-5 en `f32` sur CPU et sur GPU (wgpu ; CUDA sur la
   RTX).
2. Les tests des critères 2, 3 et 4 (`f64`, mêmes seuils, plans de P1 compris) passent avec le nouvel opérateur.
3. Critère 5 amendé refait avec le nouvel opérateur, poids initiaux de fmi : B0 sous le seuil des graines sur les six
   plans.
4. Durées rapportées (RTX CUDA, Mac Metal) ; objectif : opérateur ≤ 0,1 s par pas sur la RTX, plan complet sous les
   208 s de PyTorch `f32`.

### Résultats de P4 ter (2026-11-22)

1. **`OperateurCpr` contre `ModeleCpr`** (`tests/operateur.rs`, neuf quantités, grilles 7 × 6 × 5, 8 × 5 × 4, 16 × 12 × 9
   et 115 × 115 × 45) : ≤ 8,8e-16 en `f64` ; ≤ 4,3e-7 en `f32` sur CPU, Metal et CUDA. Tenu.
2. **Critères 2 à 4** avec le nouvel opérateur : tests passés, plans de P1 compris ; écarts du critère 4 identiques à
   ceux de l'opérateur CPU au chiffre près. Le test des pertes sur P1 demande 6 à 8 Go en `f64` sur CPU : le lancer
   avec `--test-threads 1`. Tenu.
3. **Critère 5 amendé, poids de fmi, CUDA** : B0 à 1,7 / 2,4 / 2,1 / 1,1 / 3,3 / 3,8 Hz de la graine 0, sous les seuils
   des six plans ; LNCC hors enveloppe de 1e-4 au plus sur deux plans et écart aux données dans l'enveloppe partout,
   comme avec l'opérateur CPU. Tenu.
4. **Durées** :
   - RTX, CUDA : **107–108 s par plan** (six plans), contre 240–245 s avec l'opérateur CPU et 208 s pour PyTorch
     `f32` ; démarrage à chaud 42 s, boucle jointe 61 s (0,30 s par pas, contre 0,97 s). Objectif « plan sous 208 s »
     tenu.
   - Part de l'opérateur : 13,6 s sur 200 pas (0,07 s par pas) d'après `jointe_p1`, mais sur GPU les opérations sont
     asynchrones et la mesure ne compte que ce qui s'exécute avant la première lecture ; borne haute tirée du pas
     joint moins un pas de démarrage des deux réseaux : environ 0,16 s par pas. L'objectif ≤ 0,1 s par pas n'est donc
     pas établi.
   - Mac, Metal : pas de mesure valable (Mac chargé par d'autres travaux pendant l'essai : 623 s, démarrage à chaud
     compris, deux fois plus lent qu'à vide). À refaire sur un Mac libre.

## P5 : méthode B, champ INR seul (conception, 2026-11-23)

### Ce qu'on porte

`compare_methods.estimate_inr` (référence `cb7174b`), qui fournit le `b0_init` de l'estimation jointe. Dans P1, il
est appelé par `denoise_joint.run_pipeline` **hors** des surcharges de `b0_sensitivity` : le champ garde donc
`B0_INR_CONFIG` par défaut.

- **Champ** : `INRField` comme en P4 bis, mais avec une résolution de 2 → **16** (facteur (16/2)^(1/7)), et non 32.
  Mêmes bornes `b0_bounds_hz`, même forme (tanh, centrage, RMS 1, `offset + amplitude · forme`).
- **Perte** : −LNCC(|Aᴴ(+b0)k_reg|, |Aᴴ(−b0)k_inv|) (fenêtre de 4 mm, c'est le terme `b0` de P4 bis)
  \+ **1e-2** · lissage + 1e-5 · (s² + o²). Pas d'image, pas de fidélité, pas de démarrage à chaud.
- **Optimisation** : 400 itérations d'Adam (β = 0,9 / 0,999, ε = 1e-8), lr 1e-3 (réseau) et 1e-2 (s, o), décroissance
  cosinus sur 400 itérations jusqu'à 1e-2 · lr.
- **Graine** : `INRField` fait `torch.manual_seed(0)` dans son constructeur ; le `b0_warm_start` de P1 est donc la
  graine 0, à condition que le cache de P1 date bien de `cb7174b` (vérifié par le contrôle du critère 3).

### Conception

Presque tout existe déjà ; il reste :
- `ConfigGrille::champ_b()` (16 au lieu de 32) ;
- dans `pertes`, une évaluation réduite au terme `b0`, au lissage et aux scalaires : deux passes `Aᴴ` (et `Aₜᴴ` pour le
  gradient) par itération, sans `A` ni l'image, sur l'`OperateurCpr` de P4 ter ;
- dans `jointe` (ou un module `methode_b`), la boucle de 400 pas, avec l'Adam et la décroissance cosinus déjà écrits ;
- l'exemple `methode_b_p1` (même interface que `jointe_p1`), et une option de `jointe_p1` qui prend le `b0_init` dans
  une sortie de `methode_b_p1` au lieu de P1 : la chaîne B → jointe tourne alors entièrement en Rust ;
- côté Python, `scripts/cas_jointe.py` gagne un mode `methode_b` (états du réseau du champ avant et après un pas, à
  partir des poids de fmi) et un mode `graines_b` (méthode B de PyTorch, graines 0, 1, 2 en `f64` et graine 0 en
  `f32`, sur les six plans, à lancer sur la RTX ; le `f32` reprend la mise en simple précision de
  `b0_bases_lisses.py --float32`).

### Critères (écrits avant le code)

1. **Grille du champ de la méthode B** (cas synthétique et plans de P1) : indices de table identiques à PyTorch, entier
   pour entier ; sortie du champ à ≤ 1e-12 (`f64`) / 1e-6 (`f32`) ; gradient d'une fonctionnelle linéaire aléatoire à
   ≤ 1e-10 / 1e-5.
2. **Perte et un pas** (plans de P1, `f64`, poids de fmi) :
   - perte de la méthode B à ≤ 1e-10 et son gradient par rapport à b0 à ≤ 1e-8 ;
   - paramètres après un pas d'Adam à ≤ 1e-10 de PyTorch, avec le même plancher par paramètre qu'en P4 bis (écart
     entre deux exécutions de PyTorch à nombres de fils différents) s'il est plus haut.
3. **Bout en bout, `f32` sur GPU, initialisation de Burn, graines 0, 1, 2 de Rust**, sur les six plans :
   - B0 de la graine 0 de Rust à moins du plus grand écart entre les graines 0, 1, 2 de PyTorch (RMS dans l'objet) de
     la graine 0 de PyTorch en `f64` ;
   - LNCC de la paire corrigée (critère d'évaluation de `pair_criteria`) dans l'enveloppe des graines 0, 1, 2 en
     `f64` et du calcul `f32` de PyTorch ;
   - dispersion : écarts entre graines de Rust du même ordre que ceux de PyTorch (au plus le double du plus grand), et
     écart entre les moyennes des graines de Rust et de PyTorch sous le plus grand écart entre deux graines de
     PyTorch (leçon du critère 5 de P4 bis : une seule graine ne distingue pas un biais du bruit) ;
   - contrôle : la graine 0 de PyTorch en `f64` redonne le `b0_warm_start` de P1 à la précision `f32` de son stockage
     (sinon le cache de P1 ne date pas de `cb7174b`, et on le dit).
4. **Chaîne B → jointe en Rust**, `f32` sur la RTX : la jointe partant du B0 de la méthode B de Rust (graine 0) donne
   un B0 sous le seuil des graines de la jointe (critère 5 amendé de P4 bis) et une LNCC et un écart aux données dans
   leur enveloppe. Le seuil est celui de graines à `b0_init` fixé : un dépassement ne serait donc pas forcément un
   défaut, mais il serait rapporté et discuté avant d'aller plus loin.
5. **Durées** rapportées, sans critère : méthode B et chaîne complète par plan sur la RTX, le Mac (Metal, machine
   libre) et le CPU ; référence : la méthode B de PyTorch `f32` sur la RTX, mesurée par `graines_b`.

Ordre : 1 → 2 → 3 → 4 → 5, avec les mutations qui font échouer chaque test notées. La chaîne qui va du fichier
ISMRMRD (P3) aux entrées (fenêtre de Tukey autour de l'écho, normalisation) reste à part et vient après.

### Résultat du critère 1 de P5 (2026-11-23) : grille du champ de la méthode B

`ConfigGrille::champ_b()` (facteur 8^(1/7)), `scripts/cas_inr.py p1_b` (champ seul, 115 × 115 × 45, poids tirés comme
en P4 bis), test `inr_comme_pytorch_methode_b` (`MEDOXIDE_ULF_INR_B`) : configuration du fichier égale à `champ_b()`,
géométrie des niveaux bit pour bit, indices identiques entier pour entier ; sorties ≤ 4,5e-15 et gradients ≤ 6,3e-13 en
`f64` ; en `f32` (CPU et Metal) sorties ≤ 1,4e-6 et gradients ≤ 1,7e-4, sauf le biais de la dernière couche (6,9e-2 :
la forme est centrée, ce gradient est une somme qui s'annule ; PyTorch `f32` s'écarte de 7,1e-2 sur le même cas, il
passe par l'amendement du 2026-11-22). Tenu.

### Résultat du critère 2 de P5 (2026-11-23) : non tenu tel qu'écrit sur deux plans, amendement proposé

`scripts/cas_jointe.py methode_b` (cas synthétique versionné et six plans de P1), test `tests/methode_b.rs`
(`MEDOXIDE_ULF_P1_B`) :
- terme b0 et lissage à l'état initial ≤ 1,1e-14 et 1,0e-13 ; gradient par rapport à b0 ≤ 1,6e-11 ; perte avant
  chaque pas ≤ 9,2e-14 : **tenu** partout ;
- paramètres après un et deux pas : ≤ 5e-11 sur quatre plans et sur le cas synthétique, **hors seuil sur `sub000_yz`**
  (jusqu'à 2,0e-10) **et `subB_yz`** (jusqu'à 4,1e-8 au second pas). Les planchers mesurés entre un et plusieurs
  threads de PyTorch sont ≤ 1e-11 et n'en rendent pas compte.

**Cause.** À l'initialisation de fmi, la sortie du réseau du champ est presque constante (écart-type 2e-6 pour une
moyenne de 0,12 après `tanh`) ; le centrage de `INRField.b0` amplifie environ 1e5 fois l'arrondi de chaque valeur. La
moyenne elle-même est exacte (identique à `math.fsum`) : ce sont les sorties du MLP, qui diffèrent d'un ulp entre Rust
et PyTorch (ordre des produits, `tanh`), qui donnent le b0 initial à 1,5e-11 sur les six plans. Les premiers pas d'Adam
amplifient ensuite les composantes du gradient de l'ordre de ε.

**Mesure** (`cas_jointe.py sensibilite_b`) : PyTorch contre lui-même, la sortie du réseau étant multipliée par
1 + 4 r · 2⁻⁵³ (r = ±1) — b0 initial écarté de 2,7e-11, un peu plus que Rust. Écart des paramètres après les pas :
7,6e-10 sur `sub000_yz`, 9,2e-8 sur `subB_yz`, ≤ 8,8e-11 ailleurs. Les mêmes deux plans sortent, dans le même ordre de
grandeur. Sur `subB_yz`, l'écart de Rust vaut **0,44 fois** cette sensibilité sur chacun des dix paramètres hors seuil
(même direction, amplitude proportionnelle) ; sur `sub000_yz`, 0,26 fois pour la table du niveau 7 et 1,8 fois pour le
biais de la couche 1 (sensibilité d'un seul tirage).

**Amendement proposé** (à décider) : un paramètre passe aussi s'il reste sous deux fois la sensibilité de PyTorch à
4 ulp sur la sortie du réseau (`sensibilite_pytorch/…`), en plus du plancher des threads ; le terme, le gradient et la
perte avant chaque pas gardent leurs seuils. Avec cette règle, les six plans passent (pire rapport : 0,89 du seuil
amendé).
