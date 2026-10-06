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

## 2026-10-05 — Nommer un crate d'après la méthode, et citer ses sources

- **Quoi** : `medoxide-mask` est renommé `medoxide-fetalbet` (dossier, nom du
  paquet, dépendance de `medx`, appels `medoxide_fetalbet::`) et la sous-commande
  `medx mask` devient `medx fetalbet`. On nomme le crate et la commande d'après
  l'*algorithme* utilisé et son domaine (Fetal-BET, IRM fœtale) plutôt que d'après
  la fonction générique (« mask ») : l'utilisateur sait précisément ce qui
  s'exécute, et une autre méthode de masque aurait sa propre sous-commande.
  `git mv` conserve l'historique du dossier. (Les entrées de l'étape 8 ci-dessus
  parlent de `medx mask` : c'était le nom à ce moment-là.)
- **Quoi** : l'attribution d'un travail dérivé. Fetal-BET est sous CC BY 4.0 :
  il faut citer les auteurs, donner la licence, **indiquer les modifications**
  (conversion PyTorch vers ONNX vers Burn) et ne pas laisser croire à leur
  approbation. Fait dans l'en-tête de `model.rs` et dans le `README.md`.
- **Où** : `crates/medoxide-fetalbet/src/model.rs`, `README.md`.

## 2026-10-06 — Télécharger les poids automatiquement

- **Quoi** : lire un flux réseau par blocs. La réponse HTTP (crate `ureq`) se lit
  avec le trait `Read` : `lecteur.read(&mut tampon)` remplit un tampon de 64 Ko et
  renvoie le nombre d'octets lus (0 = fin). À chaque bloc, on l'écrit dans un
  fichier (`write_all`, trait `Write`) et on alimente un hachage SHA-256
  (`Sha256::update`). Jamais plus de 64 Ko en mémoire pour 121 Mo de poids.
- **Pourquoi ici** : vérifier l'intégrité sans relire le fichier, et ne pas
  dépendre de la mémoire disponible.
- **Où** : `download_verified` dans `crates/medoxide-fetalbet/src/lib.rs`.

- **Quoi** : écrire un fichier de façon atomique. On écrit dans `attunet.part`, on
  vérifie le SHA-256, puis `std::fs::rename` le renomme en `attunet.bpk`. Si
  quoi que ce soit échoue, on supprime le `.part` : on n'obtient jamais un faux
  fichier de poids.
- **Quoi** : une fermeture appelée immédiatement, `let resultat = (|| { ... })();`,
  qui regroupe plusieurs `?` : quel que soit l'endroit de l'échec, on passe
  ensuite au nettoyage, puis on renvoie l'erreur d'origine.

- **Quoi** : `?` sur un `Option` (comme sur un `Result`) : `home.filter(..)?` fait
  sortir de la fonction avec `None` si la valeur est absente. Et
  `Option<&Path>` : `model.as_deref()` convertit un `Option<PathBuf>` en
  `Option<&Path>` sans consommer l'original.
- **Quoi** : passer l'environnement en paramètres (`cache_dir_from(xdg, home)`)
  plutôt que lire `std::env` dans la fonction : on peut la tester sans modifier
  les variables d'environnement du processus.
- **Pourquoi ici** : `segment` accepte maintenant `Option<&Path>` pour les poids ;
  `None` déclenche le téléchargement **après** avoir validé l'entrée et le dossier
  de sortie, pour qu'une faute de frappe ne coûte pas 121 Mo.

Choix de conception : l'URL est épinglée sur une révision précise du dépôt
Hugging Face, et le SHA-256 est vérifié ; un fichier déjà en cache n'est pas
revérifié (il l'a été au téléchargement, et écrit atomiquement).

**Vérifié** : faute de frappe ou `--model` inexistant : aucun téléchargement ;
réseau en panne (proxy invalide) : message clair, aucun fichier laissé ; premier
lancement : téléchargement, SHA-256 correct, masque identique (Dice 0,999987 sur
fetus_06) ; second lancement : sans réseau. Test réseau `#[ignore]` : une
empreinte fausse est refusée et ne laisse aucun fichier.

## 2026-10-06 — Extraire un crate commun : `medoxide-core`

- **Quoi** : un crate de plus dans le workspace, et le **sens des dépendances**.
  `medoxide-fetalbet` dépend de `medoxide-core` (`medoxide-core = { path =
  "../medoxide-core" }`) ; le core ne dépend d'aucun module. Un futur module (SVR)
  pourra lire des NIfTI sans tirer Burn, le modèle ni le téléchargement des poids.
- **Pourquoi ici** : c'est le premier besoin réel d'un second module (lire des NIfTI
  avec leur affine), le déclencheur prévu pour créer ce crate, pas avant.
- **Où** : `crates/medoxide-core/src/lib.rs` (`volume_info`, `read_volume`,
  `VolumeInfo`, `CoreError`) ; `Cargo.toml` racine (membres du workspace).

- **Quoi** : déplacer du code sans changer l'interface visible. `medx` n'appelle que
  `segment` et `default_model_path`, restés dans `medoxide-fetalbet` : ses utilisateurs
  ne voient rien. Les quatre tests de lecture NIfTI ont migré avec leur code (mêmes noms,
  mêmes données) ; la suite complète donne les mêmes 18 tests (15 réussis, 3 ignorés).
- **Quoi** : envelopper l'erreur d'un autre crate. `MaskError::Core(CoreError)` +
  `impl From<CoreError> for MaskError` : `?` convertit seule. Le message est repris tel
  quel (`write!(f, "{e}")`), donc les textes documentés dans le livre n'ont pas changé.
  L'écriture (`write_mask`) passe par `.map_err(CoreError::from)?` : une erreur de `nifti`
  devient d'abord une `CoreError`, puis une `MaskError`. Un seul chemin d'erreur NIfTI.

**Vérifié** : mêmes noms et statuts de tests avant et après ; messages d'erreur de
`medx fetalbet` inchangés (entrée absente, 4D, fichier invalide, dossier de sortie
absent) ; masque de fetus_06 inchangé (Dice 0,999987, 3 voxels de différence).
`write_mask` reste dans `medoxide-fetalbet` (le SVR n'en a pas encore besoin).

## 2026-10-06 — Un stack et sa géométrie (étape 1a du SVR)

- **Quoi** : `nalgebra`, des matrices de **taille fixe** (`Matrix4<f64>`, `Matrix3<f64>`).
  Les dimensions sont connues à la compilation : pas d'allocation, et le compilateur refuse
  de multiplier des matrices incompatibles. `Matrix4::from_fn(|r, c| ...)` construit une matrice
  case par case ; `affine.fixed_view::<3, 3>(0, 0)` en donne un bloc 3×3 sans copie ;
  `colonne.norm()` et `a.dot(&b)` donnent norme et produit scalaire.
- **Pourquoi ici** : l'affine voxel → monde est une matrice 4×4, et toutes les transformations
  rigides du SVR (rotation + translation par coupe) seront des produits de telles matrices.
- **Où** : `crates/medoxide-svr/src/lib.rs` ; dépendance `nalgebra = "0.35"` (`Cargo.toml`).

- **Quoi** : les **coordonnées homogènes**. On écrit un point `(i, j, k, 1)` : une seule
  multiplication par la matrice 4×4 applique rotation, mise à l'échelle **et** translation.
  Les 3 premières colonnes de la partie 3×3 sont les déplacements monde (mm) d'un pas d'indice
  sur chaque axe ; leurs normes sont les espacements.
- **Quoi** : `f64::from(x)` convertit un `f32` **sans perte**, contrairement à `as` qui peut
  tronquer ; l'en-tête NIfTI stocke l'affine en `f32`, on calcule ensuite en `f64`.

- **Quoi** : des **champs privés** pour garantir des **invariants**. `Stack` ne se construit
  que par `Stack::read`, qui vérifie la géométrie ; tout `Stack` existant est donc valide, et les
  étapes suivantes n'ont pas à la revérifier. Les accesseurs (`affine()`, `data()`, ...) rendent
  des références en lecture seule.
- **Quoi** : `!(n > 0.0)` détecte aussi `NaN` (toute comparaison avec `NaN` est fausse), ce que
  `n <= 0.0` ne ferait pas.

**Décisions de conception** (étude 01 des dépôts) : monde RAS+ du `sform`, centres de voxel aux
indices entiers, coupe `k` = plan du 3ᵉ axe. **Un déterminant négatif est accepté** (repère
d'indices « main gauche ») : les 96 stacks du jeu de développement le sont tous, l'image n'est
jamais retournée. Refusés : `sform` absent, cisaillement (cosinus entre colonnes > 1e-3), colonne
nulle ou non finie, `pixdim` incohérent avec les normes de l'affine (> 1e-3 mm). L'espacement
est la norme des colonnes de l'affine, c'est-à-dire la géométrie réelle.

**Vérifié** : 9 tests. Cas valides et d'erreur sur fichiers synthétiques (sform absent,
cisaillement, `pixdim` incohérent, affine dégénérée, 4D, fichier absent, affine à déterminant
négatif acceptée) ; affines lues pour 3 volumes de Fetal-BET égales à celles de nibabel (arrondies
à 4 décimales) ; **les 96 stacks du jeu de développement se chargent, tous à déterminant négatif**.

## 2026-10-06 — Le repère de chaque coupe et la première durée de vie (étape 1b du SVR)

- **Quoi** : une **durée de vie** (*lifetime*), notée `'a`. `Slice<'a>` ne possède pas ses données : elle
  **emprunte** un `Stack` (`stack: &'a Stack`) et le compilateur garantit qu'une coupe ne survit jamais
  à son stack (sinon elle pointerait vers de la mémoire libérée). La coupe ne contient qu'une référence et
  un indice : elle est `Copy` (`#[derive(Clone, Copy)]`).
- **Pourquoi ici** : une coupe n'est qu'un indice `k` sur le stack ; la copier avec ses voxels n'aurait
  aucun sens et coûterait cher.
- **Où** : `Slice<'a>`, `Stack::slice`, `Stack::slices` dans `crates/medoxide-svr/src/lib.rs`.

- **Quoi** : `'_`, la durée de vie *anonyme*. `fn slice(&self, k: usize) -> Option<Slice<'_>>` veut dire
  « la coupe rendue emprunte `self` » : le compilateur déduit le lien, on ne le nomme pas.
- **Quoi** : une méthode qui rend une vue de durée de vie **`'a`** (celle du stack) et non celle de `&self` :
  `fn data(&self) -> ArrayView2<'a, f32>`. On peut ainsi garder la vue après la disparition de l'objet
  `Slice` (une simple référence), tant que le stack existe.
- **Quoi** : `impl Iterator<Item = Slice<'_>>` comme type de retour : `slices()` rend un itérateur sans
  nommer son type ; `.map(move |k| ...)` capture `self` par la fermeture (`move`).
- **Quoi** : `bool::then_some(valeur)` : `(k < n).then_some(x)` donne `Some(x)` ou `None`.

**Géométrie de la coupe `k`** : `A · T(0,0,k)`, c'est-à-dire la même partie linéaire que `A` et l'origine
décalée de `k` fois la 3ᵉ colonne ; elle envoie `(i, j, 0, 1)` sur la position monde. **Normale = 3ᵉ colonne
normalisée** (pas le produit vectoriel des axes du plan, qui pointe à l'envers pour une affine à
déterminant négatif : un test synthétique le montre). Épaisseur = espacement entre coupes. Centre
géométrique = point `((nx−1)/2, (ny−1)/2)`.

**Vérifié** (13 tests) :
- critère 1 : coordonnées monde de 5 points par coupe (coins et centre) et de 2000 points aléatoires par
  stack, sur les 8 volumes de Fetal-BET et les 96 stacks du jeu de développement : écart maximal à
  nibabel **1,1e-13 mm** (seuil 1e-6) ;
- critère 2 : écart maximal à SimpleITK (converti LPS → RAS) **2,6e-4 mm**. Seuil révisé de 1e-4 à 1e-3 mm
  après diagnostic : ITK construit sa géométrie avec `pixdim` (f32) et une direction orthonormalisée
  (en recalculant ainsi l'affine de nibabel, l'écart tombe à 5,5e-5 mm), soit un trois-millième de voxel,
  alors que l'affine stockée (celle de nibabel et de `Stack`) reste la référence stricte ;
- critère 3 : invariants sur 3413 coupes de 104 stacks (normale unitaire, colinéaire à la 3ᵉ colonne et
  orthogonale au plan, pas dans le plan et entre coupes égaux à `pixdim` à 1e-4 mm) ;
- trois **mutations volontaires** du code (mauvaise colonne dans l'affine de coupe, normale par produit
  vectoriel, centre en `nx/2`) sont chacune détectée par au moins un test. Le centre géométrique n'est
  contrôlé que par un stack synthétique, pas par les références Python.

## 2026-10-06 — Lire un ensemble de stacks et les décrire (étape 1c du SVR)

- **Quoi** : **collecter des `Result`**. `chemins.iter().map(|c| Stack::read(c)).collect()` avec un type de
  retour `Result<Vec<Stack>, SvrError>` rassemble tous les succès en un `Vec`, ou s'arrête à la première erreur
  et la renvoie. Le type de retour de la fonction suffit à dire à `collect` quoi construire.
- **Pourquoi ici** : lire les 6 à 9 stacks d'un sujet d'un coup, en s'arrêtant proprement au premier fichier
  fautif.
- **Où** : `read_stacks` dans `crates/medoxide-svr/src/lib.rs`.

- **Quoi** : le **contexte d'erreur**. `SvrError::Core(...)` devient `SvrError::Read { path, source }` : sur un
  ensemble, le message doit nommer le fichier fautif. `map_err` avec une fermeture
  (`|source| SvrError::Read { path: ..., source }`) rattache le chemin à l'erreur du core. Les variantes
  d'`SvrError` connaissent donc toutes leur fichier.
- **Quoi** : une **boîte englobante** alignée sur les axes du monde (`BoundingBox`). Elle se calcule en prenant,
  pour les 8 coins du stack, le minimum et le maximum composante par composante (`Vector3::inf` / `sup`). La
  boucle `for i in [0, nx - 1]` parcourt directement un tableau de deux éléments. `union` combine deux boîtes.
- **Quoi** : une **sous-commande imbriquée** avec `clap` : `medx svr info --input a b c` (`#[command(subcommand)]`
  dans `Svr`, puis un `enum SvrAction`) et `#[arg(long, num_args = 1.., required = true)]` pour accepter
  plusieurs valeurs d'une option.

**Rappel important (étude 02)** : une boîte englobante est un indicateur **faible** de cohérence entre stacks :
deux têtes à 154 mm l'une de l'autre s'y recouvrent encore. La vraie détection viendra des barycentres de masque
(étape 1e). Ici, la boîte décrit la géométrie, elle ne décide de rien.

**Vérifié** (17 tests) : boîtes de stacks synthétiques, y compris à axe x inversé ; ordre conservé et fichier
fautif nommé par `read_stacks` ; **boîtes des 104 stacks identiques à celles que donnent les coins calculés par
nibabel** (écart 0 mm) ; lecture en un seul appel des 15 sujets du jeu de développement, la boîte de l'ensemble
contenant celle de chaque stack. Deux mutations volontaires du code (coins en `nx` au lieu de `nx − 1`, `union`
avec minimum et maximum échangés) sont chacune détectées. **`medx svr info`** : sortie identique, normales et
boîtes comprises, à un recalcul indépendant avec nibabel sur trois stacks d'un même sujet ; erreurs avec le nom
du fichier ; code de sortie 1 en cas d'échec, 2 pour des arguments invalides.

## 2026-10-06 — Le pivot de chaque coupe : masque nettoyé et barycentre (étape 1d du SVR)

- **Quoi** : les **composantes connexes** par **parcours en largeur** (*breadth-first search*). À partir d'un
  voxel du masque, on visite tous ses voisins, puis les voisins de ceux-ci, jusqu'à épuisement : c'est une
  composante. On recommence à partir d'un voxel pas encore vu, et on garde la plus grande. Le voisinage est de
  **26** voxels (faces, arêtes et coins), comme `scipy.ndimage.label` avec `np.ones((3,3,3))`.
- **Quoi** : une **file** faite d'un `Vec` et d'un indice de lecture : on ajoute à la fin les voxels à visiter,
  l'indice `lu` avance dans ce qui reste. Pas de structure spéciale. `masque.indexed_iter()` donne les
  indices avec les valeurs.
- **Quoi** : des **entiers signés pour les voisins**. `a - 1` avec `a = 0` n'existe pas en `usize` (panique) ; on
  passe par `isize`, on teste les bornes, puis on revient en `usize` pour indexer.
- **Pourquoi ici** : de petits îlots parasites (jusqu'à plusieurs % des voxels) déplacent les boîtes
  englobantes ; le barycentre y est peu sensible, mais on le calcule sur le masque propre.
- **Où** : `largest_component`, `BrainMask`, `Stack::set_brain_mask`, `Slice::brain_pivot` dans
  `crates/medoxide-svr/src/lib.rs`.

- **Quoi** : `&mut self` pour **attacher** un masque à un stack existant : `set_brain_mask(&mut self, chemin)`.
  En cas d'échec, le stack reste intact (une méthode qui consommerait `self` le perdrait). Le champ
  `mask: Option<BrainMask>` est `None` tant qu'aucun masque n'est attaché.
- **Quoi** : `Option` comme **réponse honnête**. `Slice::brain_pivot()` rend `None` sans masque : le repli éventuel
  sur le centre géométrique est une décision de l'appelant, jamais silencieuse (un pivot qui changerait de nature
  sans prévenir fausserait les paramètres de translation).
- **Quoi** : `Option::as_ref()` pour lire dans un `Option<BrainMask>` sans le déplacer, et `?` appliqué à un
  `Option` (`self.stack.mask.as_ref()?`) qui fait rendre `None` à la fonction.

**Le pivot P3** (étude 02) : le barycentre 3D du masque nettoyé, pris dans le plan de la coupe `k` (les `(i, j)` du
barycentre, `k` pour la 3ᵉ coordonnée), converti en monde. Garde-fous : le masque doit avoir la **même grille** que
le stack (dimensions, et affine à 1e-3 près, `sform` obligatoire) et ne pas être vide.

**Vérifié** (21 tests, 22 avec le test long) :
- jeu synthétique avec résultat calculé à la main : bloc de 8 voxels, un voxel qui ne le touche que par un
  **coin** (conservé : 26 voisins), un îlot (écarté), barycentre exact à 1e-12 ;
- masque vide, mauvaise affine, mauvaises dimensions, `sform` absent, fichier absent : erreurs claires, stack
  inchangé ;
- **les 96 stacks et leurs masques `medx fetalbet`** : même nombre de voxels conservés que `scipy.ndimage.label`
  (202 276 voxels écartés en tout), barycentre identique (écart 0), **3033 pivots de coupe égaux à numpy à
  2,8e-14 mm** ; le test complet est `#[ignore]` (130 s en debug), une version à un stack sur 8 tourne dans la suite ;
- trois mutations volontaires (voisinage à 6, pivot sans décalage par coupe, plus petite composante) sont
  détectées. La première ne l'est que par les tests synthétiques : sur la version rapide, les masques réels
  ne distinguent pas 6 et 26 voisins (non mesuré sur les 96).

## 2026-10-06 — Repérer les stacks qui ne partagent pas un repère (étape 1e du SVR)

- **Quoi** : les **composantes connexes d'un graphe**, par le même parcours en largeur qu'en 1d. Ici les
  « nœuds » sont les stacks et il y a un lien entre deux stacks si l'écart de leurs barycentres de masque est
  ≤ `d`. On suit les liens de proche en proche : c'est le **chaînage** (A proche de B et B proche de C mettent
  A, B et C ensemble, même si A et C sont plus éloignés).
- **Pourquoi ici** : le repère monde ne reste cohérent qu'au sein d'une série de stacks consécutifs (étude 02 :
  deux sujets sur 15 ont des groupes décalés de 24 et 154 mm). Le **groupe dominant** servira à construire le
  volume initial, sans recalage stack-à-stack au départ (principe validé).
- **Où** : `group_stacks`, `StackGroups`, `DEFAULT_GROUP_GAP_MM`, `Stack::brain_barycenter_world` dans
  `crates/medoxide-svr/src/lib.rs` ; `medx svr info --input ... --mask ...`.

- **Quoi** : **aucune information du JSON** n'intervient : géométrie et masques seulement. Le seuil
  (18 mm) est **provisoire** : plus grand écart au sein d'un groupe 13 mm, plus petit entre groupes 23 mm, soit
  10 mm de marge seulement ; fixé après avoir vu les données.
- **Quoi** : `Fn` contre `FnMut` (le compilateur l'a expliqué) : une fermeture qui **modifie** une variable de son
  environnement (un compteur) doit être `FnMut` ; `Fn` ne fait que lire. `impl FnMut(...) -> bool` en paramètre.
- **Quoi** : `sort_by_key(|g| (std::cmp::Reverse(g.len()), g[0]))` trie par taille décroissante puis plus petit
  indice ; `Vector3::push(1.0)` ajoute la coordonnée homogène (rend un `Vector4`) ; `values_mut()` donne un accès
  modifiable aux valeurs d'une table de hachage ordonnée (`BTreeMap`).
- **Quoi** : un stack sans masque donne `SvrError::NoMask`, jamais un repli silencieux.

**Vérifié** (24 tests réussis dans `medoxide-svr`, plus 2 tests longs `#[ignore]`) :
- jeu synthétique : chaînage (0–10, 10–20 liés alors que 0 et 20 sont à 20 mm pour un seuil de 15), seuil inclusif
  (à exactement 10 mm avec un seuil de 10 : liés ; avec 9,9 : séparés), ordre (plus grand groupe d'abord, puis plus
  petit indice), liste vide, symétrie de la matrice de distances ;
- **15 sujets** : groupes identiques à ceux de la référence Python (13 sujets à un seul groupe, deux sujets à deux
  groupes de tailles 4+2 et 5+4) et distances identiques à 4,4e-14 mm ;
- `medx svr info --mask` : sortie identique à la référence Python sur trois sujets ;
- trois mutations volontaires (pas de chaînage, seuil strict, plus petit groupe d'abord) sont détectées. Les deux
  premières ne le sont que par les tests synthétiques : sur les données réelles, chaînage et comparaison au point
  de départ donnent les mêmes groupes.

**Erreur de ma part, corrigée** : mon premier test comparait mal les groupes parce que le fichier de référence
liste les stacks groupe par groupe et non par indice ; je l'ai pris pour un défaut du code Rust avant de lire le test.

## 2026-10-07 — Lire un volume en un point du monde (étape 2a du SVR)

- **Quoi** : l'**interpolation trilinéaire**. La valeur en un point est la moyenne pondérée des **8 voxels voisins**,
  les poids étant les fractions de distance sur chaque axe (`1 - t` d'un côté, `t` de l'autre, sur x puis y puis z).
  Elle est **exacte pour une fonction linéaire** des indices, ce qui donne un test sans référence externe.
- **Pourquoi ici** : une coupe simulée a besoin de la valeur de la reconstruction `V` à des positions arbitraires,
  qui ne tombent presque jamais sur un centre de voxel. Choix de l'étude 04 : trilinéaire plutôt que voxel le plus
  proche (pas d'erreur d'arrondi de position, fonction continue de la pose, utile au recalage par gradient).
- **Où** : `Volume`, `Volume::sample` dans `crates/medoxide-svr/src/lib.rs`.

- **Quoi** : la **matrice inverse**. Pour passer d'un point du monde à des indices de voxel continus, on applique
  l'inverse de l'affine ; `Matrix4::try_inverse()` rend un `Option` (`None` si la matrice est singulière), que
  `Volume::new` transforme en erreur (`SingularVolumeAffine`). On vérifie aussi que tout est fini.
- **Quoi** : « dehors » est une **réponse, pas une valeur**. `sample` rend `None` hors de la zone entre les centres du
  premier et du dernier voxel de chaque axe (un des 8 voisins manquerait) : on ne fabrique pas de zéro. La
  comparaison `!(v >= a && v <= b)` est vraie aussi pour `NaN`, qui échoue à toute comparaison.
- **Quoi** : `[0, 1, 2].map(|a| ...)` construit un tableau de 3 éléments en appliquant une fermeture à chaque
  élément ; `clamp(min, max)` borne une valeur ; `Vector3::from_fn`.

**Défaut trouvé par les tests et corrigé** : un point situé exactement sur le dernier centre de voxel sortait de la
grille, car l'aller-retour monde → indices par une affine avec rotation donne `3,0000000000000004` au lieu de `3`.
`sample` accepte désormais 1e-9 voxel de tolérance au bord (`GRID_BOUNDARY_TOLERANCE`), sans sens physique, puis ramène
la coordonnée sur le bord.

**Critère 1 de l'étude 04, scindé avant la mesure** : contre `scipy.ndimage.map_coordinates` (même correspondance
monde → indices, interpolation indépendante) : écart relatif au contraste < 1e-5 ; contre SimpleITK
(`EvaluateAtPhysicalPoint`, qui valide aussi l'orientation) : < 1e-3, seuil moins strict parce qu'ITK construit sa
géométrie avec `pixdim` (écart de 2,6e-4 mm mesuré à l'étape 1b).

**Vérifié** (5 nouveaux tests) : sur 4 paires de stacks réels (volume = stack axial, points = pixels de coupes du stack
coronal du même sujet, affines obliques à déterminant négatif), **97 340 points comparés** : écart relatif au contraste
de **0** avec scipy et de **2,1e-5** avec SimpleITK, et le domaine « dans la grille » de Rust est identique à celui de la
référence, point par point (0 incohérence). Fonction linéaire exacte à 1e-5 avec une affine tournée, mise à l'échelle et
à déterminant négatif ; dehors (y compris `NaN`, 1e12, un demi-voxel avant le premier centre) donne `None` ; affine
singulière refusée ; axe d'un seul voxel. Trois mutations volontaires (poids échangés, voisin le plus proche, affine
directe à la place de l'inverse) sont détectées.

---

## 2026-10-07 — La réponse impulsionnelle d'une coupe (étape 2b du SVR)

- **Quoi** : la **PSF** (réponse impulsionnelle). Un pixel de coupe n'est pas la valeur du volume en un point : c'est une
  moyenne pondérée du volume autour du centre du pixel. On prend une **gaussienne** `σ = FWHM / 2,3548`, de FWHM
  `1,2 ×` la taille du pixel dans le plan et égale à l'épaisseur hors plan. Sur le jeu de développement : σ de 0,32 à
  0,43 mm dans le plan, 1,27 à 1,49 mm hors plan, une PSF très allongée selon la normale.
- **Quoi** : une **covariance tournée**. La gaussienne est diagonale dans le repère de la coupe (u, v, normale) ; dans
  le monde, `Σ = L · diag(σ²) · Lᵀ`, `L` ayant pour colonnes les axes de la coupe. Une erreur d'orientation ne fait pas
  planter : elle donne des coupes fausses, d'où une référence analytique indépendante du code.
- **Quoi** : une gaussienne **discrète**, c'est-à-dire des échantillons (décalage, poids) sur une grille régulière du repère
  de la coupe, tronquée par une boule, poids normalisés à une somme de 1. `Vec<PsfSample>` construit par trois boucles
  imbriquées ; `Psf::new` rend un `Result` (σ non fini ou ≤ 0 refusé, `!(s > 0.0)` attrape aussi `NaN`).
- **Où** : `Psf`, `PsfSample`, `Slice::psf` dans `crates/medoxide-svr/src/lib.rs`.

**Défaut trouvé par la mesure, avant tout réglage « à l'œil »** : mes réglages a priori (pas 1 σ, coupure 3 σ, comme
NiftyMIC) donnaient **1,5 % d'erreur** sur l'analytique (critère : 1 %), et la covariance des échantillons ne valait que
**0,93** de la continue. Deux causes distinctes, mesurées sur une grille de réglages et des volumes gaussiens de largeur
0,6 à 2,5 mm : (1) la boule 3D coupée à 3 σ emporte 7 % de la variance ; (2) un pas de 1 σ est trop grossier hors plan
(σ ≈ 1,5 mm est plus grand qu'un voxel). Réglage retenu, **choisi après avoir vu ces mesures** : pas **0,75 σ**, coupure
**4 σ**, soit 619 échantillons par pixel et une covariance à 0,9935 de la continue. Limite connue : pour des structures
plus étroites qu'un voxel (0,4 mm), l'erreur reste de 4e-2. Le coût (619 échantillons par pixel) sera à réexaminer à
l'étape 2c.

**Critère 4 de l'étude 04** : pour un volume gaussien (covariance `Σ_b` anisotrope et tournée), la convolution par la
PSF vaut exactement `√(|Σ_b| / |Σ_b+Σ_psf|) · exp(−½ (p−c)ᵀ (Σ_b+Σ_psf)⁻¹ (p−c))`. Erreur maximale rapportée au maximum :
**1,0e-3** pour les coupes axiale, coronale, sagittale, oblique et oblique à déterminant négatif (critère 1e-2). La
covariance attendue est reconstruite **dans le test** à partir de l'affine qu'il fabrique, donc une erreur d'axe dans
`Slice::psf` ne passe pas inaperçue.

**Vérifié** (7 nouveaux tests) : sur les 96 stacks réels (affines obliques, déterminant négatif), la variance de la PSF
le long de la normale et des axes du plan égale σ² à 1e-5 près ; somme des poids 1, moyenne nulle. Quatre mutations
volontaires (normale remplacée par un axe du plan, σ permutés, mauvais σ dans les échantillons, facteur 1,0 au lieu de 1,2)
sont détectées, et un test vérifie qu'une covariance fausse (axes permutés) est rejetée (écart > 5e-2).

---

## 2026-10-07 — Simuler une coupe et son adjoint (étape 2c du SVR)

- **Quoi** : un **opérateur linéaire jamais construit**. Simuler une coupe, c'est `y = A · x` (`x` : le volume ; `y` : les
  pixels). Pour un pixel, chacun des 619 échantillons de PSF tombe en un point du monde ; le trilinéaire le répartit sur 8
  voxels ; on obtient des couples (voxel, coefficient), les lignes de la matrice creuse `C`. Normalisation `A = D⁻¹ C` :
  on divise chaque ligne par sa somme `W`, ce qui donne la **partition de l'unité** (volume constant → coupe constante, y
  compris aux bords) et une **couverture** (part de la PSF qui tombe dans le volume) rendue avec la coupe.
- **Quoi** : l'**adjoint est la transposée**. `Aᵀ y` (pour la reconstruction) répartit `y_p / W_p` sur les voxels avec les
  **mêmes coefficients**. Pour que ce soit exact par construction, une seule fonction (`for_each_pixel`) génère les
  coefficients, et l'avant et l'arrière ne font que les consommer ; `Volume::sample` s'appuie lui aussi sur le même
  générateur trilinéaire (`trilinear`).
- **Quoi** : une fermeture passée en argument, `impl FnMut(usize, usize, &[([usize; 3], f64)], f64)`, appelée pour chaque
  pixel ; `FnMut` parce qu'elle modifie ce qu'elle capture (le tableau de sortie). Un `Vec` réutilisé d'un pixel à l'autre
  (`clear()` garde la capacité) évite des milliers d'allocations.
- **Où** : `Volume::simulate_slice`, `Volume::back_project`, `SimulatedSlice` dans `crates/medoxide-svr/src/lib.rs`.

**Critères** (fixés avant le code) et résultats :
- **Test de l'adjoint** `⟨Ax, y⟩ = ⟨x, Aᵀy⟩`, `x` et `y` aléatoires, 3 coupes obliques d'un stack à déterminant négatif,
  avec pixels de couverture nulle, partielle et complète : écart relatif **≤ 3,2e-7** (critère 1e-5).
- **Partition de l'unité** : volume constant 7,5 → coupe égale à 7,5 sur tous les pixels couverts (écart 0), `0` ailleurs.
- **Fonction linéaire du monde** : traverse l'opérateur au centre des pixels pleinement couverts avec un écart de **3,2e-7**
  (critère 1e-4) ; valide position, orientation et géométrie de bout en bout.
- **Gaussienne sur une vraie grille** (voxels de 0,5 mm, covariance tournée, coupe oblique) : **8,7e-3** du maximum
  (critère 2e-2, fixé pour inclure l'interpolation de la grille).
- Coupe hors du volume : couverture et valeurs nulles, l'adjoint ne projette rien.

**Vérifié par mutation** : 4 mutations sur 5 sont détectées (adjoint sans division par `W` ; avant sans normalisation ;
décalages de PSF non convertis en indices de voxel ; coefficients sans le poids de l'échantillon). La 5ᵉ (signe des
décalages inversé) n'est pas détectable : la PSF est symétrique, l'opérateur est identique (mutant équivalent).

**Coût mesuré** (release, un cœur, stack réel, coupe de 384×384) : **0,88 s** à l'aller, **0,85 s** au retour, 619
échantillons par pixel. À cette vitesse, une itération sur ~150 coupes coûterait plus de 2 min par sens : parallélisation
(`rayon`) et/ou réduction de l'échantillonnage à envisager avant l'étape 4, pas ajoutés maintenant.

---

## 2026-10-07 — Comparer notre opérateur à celui de NeSVoR (étape 2d du SVR)

- **Quoi** : valider contre une **implémentation externe exécutée, pas copiée**. `scripts/make_reference_acquisition.py`
  charge `slice_acquisition_torch` depuis un clone local de NeSVoR (MIT) et lui fait simuler des coupes réelles ; un test
  Rust (ignoré, à lancer en `--release`) compare nos coupes aux siennes. Aucune ligne de NeSVoR n'est dans le dépôt.
- **Quoi** : le vrai travail est la **traduction de conventions**, établie par lecture de son code : volume `(D, H, W)` =
  `(z, y, x)` isotrope et centré en unités de voxel ; position d'un pixel = `R · (local + t)` (la translation est appliquée
  *avant* la rotation) ; PSF sur une grille entière de voxels, orientée par `R` ; positions arrondies au voxel le plus proche ;
  normalisation par la somme des poids (seuil 1e-2). Notre repère monde RAS+ devient son repère centré en choisissant
  un volume aligné sur les axes du monde ; `t` se déduit de l'affine de la coupe en résolvant `R t = pos0 + R c_s`
  (pas par transposition, car les colonnes d'une affine réelle ne sont orthogonales qu'à 1e-3 près : 0,1 mm d'erreur).
- **Où** : `scripts/make_reference_acquisition.py`, test `simulated_slices_match_nesvor_acquisition_operator`.

**Critère 5 de l'étude 04** : corrélation des coupes simulées ≥ 0,99 (à rapporter, pas à forcer). Mesuré sur **4 sujets réels,
23 coupes, 588 800 pixels pleinement couverts des deux côtés** (volume isotrope de 0,8 mm tiré du stack axial, coupes du
stack coronal, affines obliques à déterminant négatif) :

| PSF de notre opérateur | corrélation min / médiane | écart quadratique relatif médian / max |
|---|---|---|
| σ identiques à NeSVoR (1,2067 × pixel dans le plan) | 0,9990 / 0,9997 | 1,5 % / 2,3 % |
| notre réglage (1,2 × pixel) | 0,9990 / 0,9997 | 1,5 % / 2,2 % |
| *témoin* : σ hors plan appliqué le long de u (mal orienté) | 0,9874 / 0,9946 | 6,3 % / 10,5 % |
| *témoin* : sans PSF (échantillonnage ponctuel) | 0,9799 / 0,9936 | 6,8 % / 11,5 % |

**Lecture honnête** : le critère de 0,99 est atteint, mais il est **peu discriminant** : un opérateur mal orienté ou sans PSF
passe presque le même seuil (min 0,987 et 0,980). Le discriminant utile est l'**écart quadratique relatif** : 1,5 % pour une
PSF correcte contre 6 à 7 % pour les témoins. Les 1,5 % résiduels sont compatibles avec l'arrondi au voxel le plus proche de
NeSVoR (jusqu'à 0,4 mm, de l'ordre de σ dans le plan) et sa PSF tronquée à 9 × 3 × 3 voxels ; je n'ai **pas** isolé cette cause
par une expérience. Les deux réglages de σ dans le plan (1,2 et 1,2067) sont indiscernables.

**Coût comparé** : NeSVoR sur CPU (torch, plusieurs fils) : 0,1 s pour un bloc de 160 × 160 pixels, soit ≈ 4 µs par pixel ;
notre opérateur sur un cœur : ≈ 6 µs par pixel. Le coût mesuré en 2c est donc du même ordre que celui de NeSVoR sur CPU ;
NeSVoR gagne surtout par le GPU (noyau CUDA) et le parallélisme.

---

## 2026-10-07 — Le trilinéaire en tenseurs Burn (étape 3a, sous-étape 1 du SVR)

- **Quoi** : un **tenseur Burn**. Dans la version utilisée (0.22 pré-version), `Tensor<D>` est un tableau de **rang `D`**
  fixé à la compilation, de type flottant par défaut (`Tensor<D, Int>` pour des entiers). Il n'a plus de paramètre de backend :
  c'est un objet **`Device`** qui choisit où le calcul se fait, à l'exécution (`Device::flex()` : CPU en Rust pur, utilisé par
  les tests ; `wgpu` : GPU). Tous les tenseurs d'un calcul vivent sur le même `Device`. (`Device::ndarray()` et `to_vec` sont
  dépréciés dans cette version : `flex` et `try_to_vec`.)
- **Quoi** : écrire un calcul **pour qu'on puisse le dériver**. L'interpolation trilinéaire de la sous-étape 2a est refaite en
  opérations de tenseurs : `floor` donne l'indice du voxel (non différentiable, ce sont des entiers) ; la fraction
  `t = c − floor(c)` (différentiable) porte le gradient par rapport à la **position**, comme dans `grid_sample` de PyTorch ;
  `select(0, indices)` lit les voxels et transmet le gradient aux **valeurs** du volume. Les 8 coins se traitent par une boucle
  sur les bits de `0..8`.
- **Quoi** : un tenseur ne répond pas `None`. La fonction rend **deux tenseurs** : les valeurs et un masque 0/1 `inside` ;
  les points dehors sont ramenés au bord (`clamp`) pour que la lecture reste valide et leur valeur est ignorée par le masque.
- **Où** : `crates/medoxide-svr/src/diff.rs` (`trilinear_sample`, `TrilinearSample`) ; première dépendance `burn` de
  `medoxide-svr` (backend `flex` en dev-dépendance, pas d'autodiff avant la sous-étape où on dérive).

**Critère** (fixé avant le code) : mêmes valeurs que `Volume::sample` à 1e-5 et même « dans la grille », sur des points tirés
au hasard dedans et dehors. **Mesuré** : écart max **9,8e-7** (grille 22×20×18, 310 points dedans), 3,4e-7 (axe d'un seul voxel),
1,2e-7 (2×3×2). Fonction linéaire exacte à 1e-4 ; centres du premier et du dernier voxel dedans, valeur exacte. Écart attendu :
`Volume` est en `f64`, Burn en `f32`.

**Défaut de mon premier test, corrigé avant de conclure** : sur les grilles à un axe d'un seul voxel et les très petites grilles,
**aucun point** tiré au hasard n'était dedans (0 sur 400), donc le test ne vérifiait pas le cas qu'il prétendait vérifier. Les
points sont maintenant tirés pour moitié dans la grille (200 dedans sur 400).

**Vérifié par mutation** (4 sur 4 détectées) : poids `t` et `1−t` échangés ; pas d'indice aplati échangés ; voisin non borné (axe
d'un seul voxel) ; masque sans borne supérieure.

---

*(à compléter à la prochaine étape : étape 3a, pose de la coupe et coût NCC en tenseurs)*
