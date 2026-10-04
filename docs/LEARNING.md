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
  `medoxide-mask` et `medx`) qui partagent un seul `Cargo.lock`, donc des
  versions de dépendances cohérentes entre eux.
- **Pourquoi ici** : on anticipe que chaque module (masquage, recalage,
  reconstruction...) sera un crate séparé, pour pouvoir les faire évoluer
  et les tester indépendamment, tout en gardant une seule commande `medx`
  en façade.
- **Où** : `Cargo.toml` (racine).

- **Quoi** : séparer une *bibliothèque* (`medoxide-mask`, qui contient la
  logique) d'un *binaire* (`medx`, qui ne fait que l'interface en ligne
  de commande et appelle la bibliothèque).
- **Pourquoi ici** : la logique de segmentation doit pouvoir être testée
  et réutilisée (par d'autres sous-commandes, ou plus tard par un
  éventuel binding) sans dépendre du CLI.
- **Où** : `crates/medoxide-mask/src/lib.rs` vs `crates/medx/src/main.rs`.

---

## 2026-10-03 — Lire l'en-tête d'un NIfTI (étape 3a)

- **Quoi** : une *crate* externe (`nifti`) ajoutée comme dépendance. On la
  déclare une fois dans `[workspace.dependencies]` (racine), et chaque crate
  membre l'hérite avec `nifti = { workspace = true }`.
- **Pourquoi ici** : lire du NIfTI (y compris `.nii.gz`) est un problème déjà
  résolu ; le réécrire n'apporterait rien au projet.
- **Où** : `Cargo.toml` (racine), `crates/medoxide-mask/Cargo.toml`.

- **Quoi** : `Result<T, E>` et l'opérateur `?`. `?` après un appel falsifiable
  veut dire « si erreur, quitte la fonction en la renvoyant ; sinon, donne-moi
  la valeur ». Il appelle au passage `From` pour convertir l'erreur dans le
  type d'erreur de *notre* fonction.
- **Pourquoi ici** : ouvrir un fichier peut échouer (absent, invalide) ; on ne
  veut pas de `panic!` mais une erreur que l'appelant peut traiter.
- **Où** : `volume_info` dans `crates/medoxide-mask/src/lib.rs`.

- **Quoi** : une variante d'enum qui *contient* une valeur
  (`MaskError::Nifti(nifti::NiftiError)`) et `impl From<NiftiError> for
  MaskError`, qui dit à Rust comment passer d'un type d'erreur à l'autre.
- **Pourquoi ici** : pour que `?` fonctionne directement sur les appels de la
  crate `nifti` tout en gardant un seul type d'erreur public.
- **Où** : `MaskError` dans `crates/medoxide-mask/src/lib.rs`.

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
  `crates/medoxide-mask/src/lib.rs`.

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
- **Où** : `crates/medoxide-mask/src/model.rs`.

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
- **Où** : `lire_f32` dans les tests de `crates/medoxide-mask/src/lib.rs`.

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
- **Où** : `read_volume` dans `crates/medoxide-mask/src/lib.rs` ; dépendance
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
- **Où** : `normalize_slices` dans `crates/medoxide-mask/src/lib.rs`.

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

---

*(à compléter à la prochaine étape : rééchantillonnage à 1 mm (5c))*
