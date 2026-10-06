# Journal d'apprentissage

Une entrée par concept Rust/Burn nouveau, au moment où il entre dans le
code. Objectif : pouvoir relire ce fichier dans six mois (ou le faire lire
à quelqu'un qui rejoint le projet) et comprendre le raisonnement, pas
seulement le résultat.

Format suggéré par entrée : **quoi** (le concept) · **pourquoi ici**
(le besoin concret qui l'a motivé) · **où** (fichier/fonction concernés).

---

## 2026-10-01 — Démarrage du workspace

- **Quoi** : un *workspace* Cargo regroupe plusieurs crates (ici
  `medoxide-fetalbet` et `medx`) qui partagent un seul `Cargo.lock`, donc des
  versions de dépendances cohérentes entre eux.
- **Pourquoi ici** : on anticipe que chaque module (masquage, recalage,
  reconstruction...) sera un crate séparé, pour pouvoir les faire évoluer
  et les tester indépendamment, tout en gardant une seule commande `medx`
  en façade.
- **Où** : `Cargo.toml` (racine).

- **Quoi** : séparer une *bibliothèque* (`medoxide-fetalbet`, qui contient la
  logique) d'un *binaire* (`medx`, qui ne fait que l'interface en ligne
  de commande et appelle la bibliothèque).
- **Pourquoi ici** : la logique de segmentation doit pouvoir être testée
  et réutilisée (par d'autres sous-commandes, ou plus tard par un
  éventuel binding) sans dépendre du CLI.
- **Où** : `crates/medoxide-fetalbet/src/lib.rs` vs `crates/medx/src/main.rs`.

---

## 2026-10-03 — Lire l'en-tête d'un NIfTI (étape 3a)

- **Quoi** : une *crate* externe (`nifti`) ajoutée comme dépendance. On la
  déclare une fois dans `[workspace.dependencies]` (racine), et chaque crate
  membre l'hérite avec `nifti = { workspace = true }`.
- **Pourquoi ici** : lire du NIfTI (y compris `.nii.gz`) est un problème déjà
  résolu ; le réécrire n'apporterait rien au projet.
- **Où** : `Cargo.toml` (racine), `crates/medoxide-fetalbet/Cargo.toml`.

- **Quoi** : `Result<T, E>` et l'opérateur `?`. `?` après un appel falsifiable
  veut dire « si erreur, quitte la fonction en la renvoyant ; sinon, donne-moi
  la valeur ». Il appelle au passage `From` pour convertir l'erreur dans le
  type d'erreur de *notre* fonction.
- **Pourquoi ici** : ouvrir un fichier peut échouer (absent, invalide) ; on ne
  veut pas de `panic!` mais une erreur que l'appelant peut traiter.
- **Où** : `volume_info` dans `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : une variante d'enum qui *contient* une valeur
  (`MaskError::Nifti(nifti::NiftiError)`) et `impl From<NiftiError> for
  MaskError`, qui dit à Rust comment passer d'un type d'erreur à l'autre.
- **Pourquoi ici** : pour que `?` fonctionne directement sur les appels de la
  crate `nifti` tout en gardant un seul type d'erreur public.
- **Où** : `MaskError` dans `crates/medoxide-fetalbet/src/lib.rs`.

**Critère 3a atteint** : dimensions et espacement identiques à nibabel sur les
8 volumes (test `volume_info_matches_nibabel`).
## 2026-10-03 — L'affine et `Option` (étape 3b)

- **Quoi** : l'*affine* d'un volume NIfTI, matrice 4×4 qui convertit un indice
  de voxel `(i, j, k)` en coordonnées en mm dans l'espace du patient
  (diagonale = espacement, hors-diagonale = rotation, dernière colonne =
  origine). On la lit dans le `sform` de l'en-tête, comme nibabel.
- **Pourquoi ici** : les volumes de test sont *obliques* (coupes non alignées
  sur les axes du scanner, orientations LPS, LIP, PIR...). Le prétraitement
  et le ré-échantillonnage inverse (étapes 5 à 7) en dépendront.
- **Où** : champ `affine` de `VolumeInfo`, `volume_info` dans
  `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : `Option<T>`, soit `Some(valeur)`, soit `None`. Rust n'a pas de
  `null` : une valeur possiblement absente est typée comme telle et le
  compilateur impose de traiter les deux cas.
- **Pourquoi ici** : si `sform_code == 0`, il n'y a pas de `sform` à lire ;
  `affine` vaut alors `None`. Le `qform` n'est pas géré (il demanderait la
  dépendance `nalgebra`) : on l'ajoutera si un volume l'exige.
- **Où** : `VolumeInfo::affine`.

**Critère 3b atteint** : affine identique à nibabel (tolérance 1e-3) sur 3
volumes d'orientations différentes ; les 8 volumes ont un `sform`.

## 2026-10-04 — Importer le modèle dans Burn (étape 4a)

- **Quoi** : `burn-onnx` (outil `onnx2burn`) convertit un fichier ONNX en code
  Rust (`attunet.rs`, une struct `Model` avec une méthode `forward`) et en
  poids au format `burnpack` (`.bpk`). `burn-import` est l'ancien nom, marqué
  « legacy ».
- **Pourquoi ici** : on récupère les poids existants de Fetal-BET sans réécrire
  l'architecture. On lance l'outil à la main et on commite le code généré
  (20 Ko, lisible) ; ce n'est pas une dépendance du projet. Les poids (121 Mo)
  restent hors de Git.
- **Où** : `crates/medoxide-fetalbet/src/model.rs`.

- **Quoi** : un *tenseur* Burn (`Tensor<4>` : le `4` est le rang, c'est-à-dire
  le nombre d'axes : lot, canaux, hauteur, largeur) et un *`Device`* (la
  machine qui exécute les calculs). Dans Burn 0.22, le backend se choisit par
  une *feature* de Cargo (`wgpu` ici, qui utilise Metal sur Mac), pas par un
  paramètre de type.
- **Pourquoi ici** : le backend wgpu est le choix du projet ; `Device::default()`
  désigne le périphérique par défaut du backend activé.
- **Où** : `burn = { features = ["std", "wgpu"], default-features = false }`
  dans `Cargo.toml` (racine) ; `Model::forward(&self, Tensor<4>) -> Tensor<4>`.

Burn n'a pas de version stable (`0.22.0-pre.4`) : on l'épingle exactement.

## 2026-10-04 — Première inférence en Rust (étape 4b)

- **Quoi** : lire des octets bruts en nombres. `std::fs::read` donne un
  `Vec<u8>` ; `chunks_exact(4)` le découpe par paquets de 4 octets ;
  `f32::from_le_bytes` convertit chaque paquet (little-endian) en `f32`.
- **Pourquoi ici** : les références Python (entrée et logits d'une tuile) sont
  des `f32` bruts : pas besoin d'une crate pour lire du `.npy`.
- **Où** : `lire_f32` dans les tests de `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : fabriquer un `Tensor` depuis un `Vec<f32>` :
  `TensorData::new(vec, [1, 1, 256, 256])` joint les valeurs à une forme, puis
  `Tensor::from_data(data, &device)` les copie sur le périphérique (GPU). À la
  sortie, `into_data().try_to_vec::<f32>()` les ramène vers le CPU.
- **Pourquoi ici** : pour passer une tuile réelle à `Model::forward` et
  comparer les logits à ceux d'onnxruntime.
- **Où** : test `inference_matches_onnxruntime`.

**Critère 4 atteint** sur la tuile de `fetus_03` (coupe 24, 4 821 voxels de
masque) : écart relatif des logits 2,3e-6 (seuil 1e-4), argmax identique sur
les 65 536 voxels, backend wgpu.

**Leçon sur le prétraitement officiel** : la normalisation de Fetal-BET divise
par l'écart-type des voxels > 0 *sans soustraire la moyenne* (ce n'est pas un
z-score), et le rééchantillonnage à 1 mm est indispensable : sur une coupe à
1,17 mm non rééchantillonnée, le modèle ne détecte aucun voxel de masque.

## 2026-10-04 — Lire les voxels d'un NIfTI (étape 5a)

- **Quoi** : `Array3<f32>` (crate `ndarray`), tableau à 3 axes dont le nombre
  d'axes est vérifié à la compilation. Indexation `a[[x, y, z]]`, `a.dim()`
  pour les dimensions, `a.iter()` pour parcourir toutes les valeurs dans
  l'ordre logique `[x, y, z]`. La crate `nifti` renvoie un tableau à nombre
  d'axes *dynamique* ; `into_dimensionality::<Ix3>()` le convertit, et peut
  échouer (d'où `MaskError::NotVolume3D`, via `map_err` qui transforme une
  erreur en une autre).
- **Pourquoi ici** : les étapes suivantes (normalisation, rééchantillonnage,
  tuiles) travaillent sur le volume en mémoire.
- **Où** : `read_volume` dans `crates/medoxide-fetalbet/src/lib.rs` ; dépendance
  `ndarray = "0.17"` (même version que `nifti`, un seul `ndarray` compilé).

- **Quoi** : l'*ordre mémoire* d'un tableau. NIfTI stocke colonne par colonne
  (Fortran), le C ligne par ligne. L'indexation et `iter()` donnent le même
  résultat logique dans les deux cas ; seule la performance change.
- **Pourquoi ici** : pour savoir pourquoi la comparaison au fichier de
  référence (écrit en ordre C par numpy) n'a pas besoin de réordonner.

**Critère 5a atteint** : voxels identiques bit à bit à `nibabel.get_fdata()` sur
les 8 volumes (données `uint16`, facteur d'échelle `NaN` = pas de mise à
l'échelle, géré correctement par `nifti`).

## 2026-10-04 — Normalisation par coupe (étape 5b)

- **Quoi** : `&mut`, l'emprunt mutable. `normalize_slices(volume: &mut
  Array3<f32>)` ne prend pas possession du volume : elle le modifie en place
  le temps de l'appel, puis l'appelant le retrouve modifié. Pas de copie de
  13 Mo, et le compilateur interdit toute autre lecture ou écriture simultanée.
- **Pourquoi ici** : le prétraitement enchaîne plusieurs étapes sur un gros
  tableau ; on évite de le recopier à chaque étape.
- **Où** : `normalize_slices` dans `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : parcourir des coupes avec `axis_iter_mut(Axis(2))` (une vue 2D
  mutable par coupe, sans copie), calculer avec des itérateurs (`filter`,
  `map`, `sum`) et modifier avec `mapv_inplace` et une *fermeture*
  (`|v| ...`, une fonction anonyme).
- **Pourquoi ici** : la normalisation s'applique séparément à chaque coupe `z`.
- **Où** : même fonction.

Détail de numérique : l'écart-type de la référence est celui de `torch.std`
(`n - 1`), pas celui de `numpy.std` (`n`, par défaut). La différence est de
8e-6 en relatif. On somme en `f64` puis on divise en `f32`.

**Critère 5b atteint** : écart relatif 0 (identique bit à bit) à la référence
MONAI sur les 8 volumes, sans rééchantillonnage (seuil visé : 1e-4).

## 2026-10-04 — Rééchantillonnage à 1 mm (étape 5c)

- **Quoi** : interpolation linéaire séparable. Le voxel de sortie `j` lit
  l'entrée à la position `j / zoom` ; avec `i0 = floor(p)` et `w = p - i0`, la
  valeur est `(1 - w) × a[i0] + w × a[i0 + 1]`, un voisin hors du volume
  comptant pour 0. En 2D on l'applique axe par axe (x puis y).
- **Pourquoi ici** : le modèle est très sensible à la résolution (sur une coupe
  à 1,17 mm non rééchantillonnée, il ne détecte rien). Fetal-BET rééchantillonne
  à 1 mm dans le plan avant l'inférence.
- **Où** : `resample_axis`, `resample_in_plane` dans
  `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : `index_axis(Axis(k), i)` (la tranche `i` selon l'axe `k`, vue sans
  copie), `scaled_add(alpha, &autre)` (`self += alpha × autre` sur toute une
  tranche) et les conversions explicites `as` (Rust ne convertit jamais
  `usize`, `f64`, `f32` implicitement).
- **Pourquoi ici** : une tranche de sortie se construit avec deux
  `scaled_add`, sans boucle sur les voxels.
- **Où** : `resample_axis`.

Règle de taille de MONAI : `round((N - 1) × zoom + 1)`, depuis l'étendue entre
les centres du premier et du dernier voxel (et non `N × zoom`).

**Critère 5c, en deux temps** :
1. Rééchantillonnage seul : écart relatif à MONAI ≤ 8e-6 sur les 8 volumes
   (seuil 1e-4) : atteint.
2. Prétraité (rééchantillonné puis normalisé) : écart relatif 0,9e-3 à 3,8e-3
   (seuil initial 1e-4 non atteint, seuil provisoire 5e-3). Cause : MONAI
   interpole en 3D et laisse des valeurs ~1e-15 là où le résultat exact est 0 ;
   le masque `> 0` de la normalisation les compte, ce qui baisse l'écart-type
   de chaque coupe d'environ 0,2 %. Ma normalisation appliquée à la sortie
   MONAI du rééchantillonnage reproduit son prétraitement à 1e-7. Effet sur le
   modèle (fetus_03) : 22 voxels d'argmax différents sur 2,9 millions, Dice
   moyen par coupe 0,9997. Le juge final est le Dice des étapes 6 et 7.

## 2026-10-04 — Plan de fenêtres glissantes (étape 6a)

- **Quoi** : l'arithmétique sur `usize` (entier non signé) : `a - b` provoque
  une *panique* si `b > a`. `saturating_sub` s'arrête à 0 à la place, ce qui
  reproduit le `max(..., 0)` de MONAI. On rencontre aussi `div_ceil` (division
  arrondie au supérieur), `find` (premier élément d'un itérateur qui satisfait
  une condition) et `map_or` (valeur par défaut si rien n'est trouvé).
- **Pourquoi ici** : le découpage d'un axe en fenêtres de 256 avec 50 % de
  recouvrement est un calcul d'indices ; une soustraction négative ne doit
  jamais passer inaperçue.
- **Où** : `window_plan` et `AxisWindows` dans `crates/medoxide-fetalbet/src/lib.rs`.

**Critère 6a atteint** : positions de départ, complétion et longueur complétée
identiques à `dense_patch_slices` de MONAI pour 9 tailles (100, 200, 240, 256,
257, 260, 300, 400, 513), dont les cas limites.

## 2026-10-04 — Inférence par tuiles (étape 6b)

- **Quoi** : les sous-fenêtres de `ndarray`. `a.slice(s![sx..sx + 256, sy..sy + 256, z])`
  est une *vue* (sans copie) sur une fenêtre ; `slice_mut` en donne une
  modifiable, et `cible += &tuile` additionne terme à terme dans la sortie.
  `a.zip_mut_with(&b, |x, &y| ...)` applique une fermeture sur les couples
  d'éléments (ici : diviser par le nombre de fenêtres de chaque voxel) ;
  `Zip::from(&a).and(&b).map_collect(...)` fait de même en produisant un
  nouveau tableau (argmax).
- **Pourquoi ici** : SliceInferer additionne les logits de chaque tuile à la
  bonne place d'un tableau de sortie, puis moyenne, puis rogne la complétion.
- **Où** : `infer_logits`, `argmax_mask` dans `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : `#[ignore]`, qui exclut un test de `cargo test` sauf avec
  `-- --ignored`. On sépare un test rapide (fetus_06) d'un test complet.
- **Pourquoi ici** : le test complet dure 27 min (176 passages par volume).
- **Où** : `tiled_inference_matches_monai_fast` / `_all`.

**Critère 6b atteint** : entrée MONAI : Dice 1,00000 sur les 8 volumes, écart
relatif des logits de fetus_03 : 3,0e-6 (seuil 1e-4). Entrée prétraitée en
Rust : Dice entre 0,99988 et 1,00000 contre le masque MONAI à 1 mm (seuil
0,99). Mesure de durée : 0,57 s par tuile (GPU, build debug), soit ~100 s par
volume de 260 ou 300 ; à reprendre à l'étape 8.

## 2026-10-04 — Retour à la grille d'origine et écriture (étape 7)

- **Quoi** : une *fermeture* qui capture son environnement. Dans
  `logits_to_mask`, `let canal = |c: usize| { ... logits ... spacing ... }` est
  une fonction anonyme qui utilise les variables définies autour d'elle, sans
  qu'on les lui passe ; on s'en sert pour traiter les deux canaux de logits
  avec le même code.
- **Pourquoi ici** : le rééchantillonnage inverse s'applique à chaque canal.
- **Où** : `logits_to_mask` dans `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : généraliser une fonction existante. `resample_axis` prend
  maintenant la longueur de sortie et le pas en paramètres : le rééchantillonnage
  à 1 mm (pas `1 / zoom`) et son inverse (pas `zoom`) partagent le même code.
- **Où** : `resample_axis`.

- **Quoi** : écrire un NIfTI avec un en-tête de référence
  (`WriterOptions::new(chemin).reference_file(&entrée).write_nifti(&tableau)`).
  L'affine, l'espacement et l'orientation sont copiés de l'entrée ; seuls les
  dimensions, le type (`uint8`) et le facteur d'échelle changent. `.nii.gz`
  active la compression.
- **Où** : `write_mask`.

Règle d'inversion de MONAI (`Invertd`) : on rééchantillonne les **logits** (pas
le masque) vers la grille d'origine, l'entrée `i` lisant la position
`i × zoom`, puis on prend l'argmax. Interpoler le masque puis seuiller à 0,5 donne
un résultat différent (Dice 0,991 sur fetus_03).

**Critère 7 atteint** :
- 7a : 0 voxel de différence avec les masques officiels sur les 8 volumes.
- 7b : dimensions, espacement et affine identiques à l'entrée, masque relu à
  l'identique ; fichier relu aussi par nibabel (uint8, affine égale, `sform_code`
  conservé, voxels identiques aux masques officiels).

## 2026-10-04 — Brancher `segment` et la commande `medx mask` (étape 8)

- **Quoi** : un code de sortie propre. `main` renvoie un `ExitCode` (0 si tout
  va bien, 1 sinon) au lieu d'afficher l'erreur et de sortir avec 0.
- **Pourquoi ici** : un outil en ligne de commande est utilisé dans des
  scripts ; une erreur qui sort avec le code 0 passe inaperçue.
- **Où** : `main` dans `crates/medx/src/main.rs`.

- **Quoi** : une option `clap` avec variable d'environnement et valeur par
  défaut : `#[arg(long, env = "MEDOXIDE_MODEL", default_value = "models/attunet.bpk")]`.
  Priorité : ligne de commande, puis variable, puis défaut. La fonctionnalité
  `env` de `clap` doit être activée dans `Cargo.toml`.
- **Pourquoi ici** : les poids (121 Mo) ne sont pas dans Git et peuvent
  se trouver n'importe où.
- **Où** : `Command::Mask::model`.

- **Quoi** : un test de bout en bout qui chaîne tout (`.nii.gz` vers `.nii.gz`)
  contre les masques officiels, et un compilé en `--release` pour mesurer le
  temps.
- **Où** : `segment_end_to_end_fast` (fetus_06) et `segment_end_to_end_all`
  (`#[ignore]`) dans `crates/medoxide-fetalbet/src/lib.rs`.

**Critère 8 atteint** : `medx mask` (build release) produit les 8 masques, code
de sortie 0. Contre les masques de Fetal-BET : Dice entre 0,999899 et 1,000000
(0 à 14 voxels différents sur 2,6 à 3,3 millions), affine identique, `uint8`.
Temps par volume (release, GPU Apple) : 25 s pour fetus_06 (une fenêtre par
coupe), 90 à 112 s pour les autres (176 passages de 256×256 pour les tailles 260
et 300). Pas d'optimisation faite (décision : mesurer seulement) : le passage en
release ne change presque rien par rapport au debug (~100 s), ce qui suggère
que le temps est dominé par le GPU ou la lecture synchrone des résultats.

---

*(à compléter à la prochaine étape : (à définir : optimisation du temps d'inférence, autres modules))*
