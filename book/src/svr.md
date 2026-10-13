# medx svr : reconstruction coupe-vers-volume

> **En développement.** `medx svr recon` fonctionne de bout en bout et a été évalué
> ([Validation](validation.md#svr-reconstruction-coupe-vers-volume)), mais sur un
> seul jeu de données et sans vérité terrain sur des acquisitions réelles. Lisez les
> [limites](#limites) avant de l'utiliser.

En IRM fœtale, on acquiert plusieurs **stacks** de coupes 2D épaisses (par exemple un
stack axial, un coronal et un sagittal). Le fœtus bouge entre deux coupes : chaque
coupe est donc dans une position légèrement différente. `medx svr recon` estime le
mouvement de chaque coupe, puis reconstruit **un volume 3D** à partir de toutes les
coupes. Les méthodes sont décrites dans
[Algorithmes et références](algorithmes.md#svr-reconstruction-coupe-vers-volume).

## Reconstruire un volume

Il faut **au moins deux stacks de plans différents** et **un masque cérébral par
stack**. Les masques s'obtiennent avec [`medx fetalbet`](fetalbet.md) :

```bash
for plan in ax cor sag; do
  ./target/release/medx fetalbet --input stack_$plan.nii.gz --output masque_$plan.nii.gz
done

./target/release/medx svr recon \
  --input stack_ax.nii.gz stack_cor.nii.gz stack_sag.nii.gz \
  --mask  masque_ax.nii.gz masque_cor.nii.gz masque_sag.nii.gz \
  --output volume.nii.gz --poses poses.tsv
```

La commande écrit sur la sortie d'erreur le nombre de stacks et de coupes, l'algorithme et
le device de calcul, puis `Volume écrit dans "…" en N s`.

| Option | Rôle | Défaut |
|---|---|---|
| `--input` | Deux stacks NIfTI au moins (`.nii` ou `.nii.gz`). | requis |
| `--mask` | Un masque cérébral par stack, **dans le même ordre**, sur la même grille que le stack. | requis |
| `--output` | Volume reconstruit (NIfTI `float32`, compressé si le nom finit par `.gz`). Son dossier doit exister. | requis |
| `--poses` | Écrit aussi le mouvement estimé de chaque coupe ([format](#fichier-de-poses)). | aucun |
| `--motion` | Estimation du mouvement : `s2s` (coupe-coupe) ou `s2v` (coupe-volume). | `s2s` |
| `--device` | Où se font les calculs : `cpu` ou `wgpu` (GPU). | CPU, voir [ci-dessous](#cpu-ou-gpu) |
| `--resolution-mm` | Côté des voxels du volume reconstruit. | 0,8 |
| `--alpha` | Poids de la régularisation (lissage) : plus il est grand, plus le volume est lisse. Trop faible, le volume garde un hachurage régulier et des valeurs négatives ; trop fort, il devient flou. L'optimum dépend du sujet (≈ 0,05 pour un sujet calme, ≥ 0,3 pour un sujet très agité). | 0,1 |
| `--min-stack-coverage` | Nombre minimal de stacks qui doivent atteindre un voxel pour qu'il soit reconstruit. Avec 2 (défaut), les zones vues par un seul stack, souvent des coupes mal placées qui forment des « languettes » hors du cerveau, ne sont pas reconstruites ; 1 : tous les voxels touchés. | 2 |
| `--edge-kappa` | **Expérimental.** Reconstruction préservant les contours (Huber, trois repondérations) : seuil relatif κ, un contour étant une différence entre voxels voisins supérieure à κ fois la différence médiane ; 0 : lissage de Tikhonov seul. Sur données réelles, aucun gain démontré (voir [Validation](validation.md#régularisation-préservant-les-contours)). | 0 |
| `--s2s-sweeps` | Avec `s2s` : nombre de balayages de toutes les coupes. | 3 |
| `--s2v-cycles` | Avec `s2v` : nombre de cycles « recaler puis reconstruire ». | 6 |

## Choisir l'estimation du mouvement

| `--motion` | Principe | Durée | À retenir |
|---|---|---|---|
| `s2s` (défaut) | Chaque coupe est recalée contre les coupes des **autres stacks** qui la croisent, sans reconstruire de volume intermédiaire. | environ 2,5 minutes pour 3 stacks (98 coupes), sur un Mac à 12 cœurs | Rapide. Sur nos six sujets, la qualité de la reconstruction est comparable à celle de `s2v`. |
| `s2v` | Chaque coupe est recalée contre une référence calculée à partir des **autres** coupes (elle ne contient pas la coupe recalée), en 6 cycles ; le volume n'est reconstruit qu'à la fin, avec les poses finales. | environ 2,5 minutes pour 3 stacks (98 coupes) sur un Mac à 12 cœurs | Plus lent. Méthode classique de l'état de l'art (coupe-volume). |

Les deux méthodes produisent les mêmes sorties (une transformation rigide par coupe),
et la reconstruction du volume est la même.

## CPU ou GPU

Par défaut, les calculs se font sur **le CPU de la machine**. Le choix du device est fait **par
algorithme** : pour l'instant, c'est le CPU pour les deux. Le recalage coupe par coupe lance un
très grand nombre de petits calculs, que le GPU (`--device wgpu`, Metal sur Mac) traite
**plus lentement** que le CPU, avec des poses de même qualité :

| Mesure | CPU | GPU (wgpu) |
|---|---|---|
| `s2s`, stacks simulés | 18 s | 118 s |
| `s2s`, un sujet réel (3 stacks) | 150 s | 556 s |
| `s2v`, un sujet réel (3 stacks) | 151 s | 349 s (recalage par lots) |

Les poses obtenues sur GPU diffèrent un peu de celles du CPU sur quelques coupes (calcul en
`f32` sur GPU : le minimum local trouvé n'est pas toujours le même), sans être moins bonnes.
Sur le CPU, le résultat est **reproductible** : deux exécutions donnent exactement les mêmes poses.
Pour le coupe-volume, le calcul sur GPU coupe par coupe a été interrompu après 90 minutes ; avec `--device wgpu`, la commande recale les coupes **par lots** (toutes les coupes d'un stack dans un seul calcul), ce qui ramène la durée à 349 s, toujours plus que les 151 s du CPU.

## Fichier de poses

Avec `--poses`, un fichier texte est écrit : **une ligne par coupe**, de la forme

```text
stack<TAB>coupe<TAB>m11<TAB>m21<TAB>m31<TAB>m41<TAB>m12 … m44
```

`stack` et `coupe` sont comptés à partir de 0 (le stack est repéré par sa position dans `--input`).
Les 16 valeurs sont la matrice 4 × 4 **en ordre colonne**, en coordonnées monde (mm, RAS+) : le
mouvement rigide de la coupe, composé **après** l'affine de son en-tête. En Python :

```python
m = np.array([float(v) for v in champs[2:]]).reshape(4, 4, order="F")
```

Les nombres sont écrits avec assez de chiffres pour être relus exactement.

## Décrire des stacks : `medx svr info`

Avant de reconstruire, `medx svr info` affiche la géométrie de chacun des stacks (dimensions,
espacement, normale des coupes, boîte dans le monde) et, avec des masques, repère les stacks qui
ne partagent pas un repère (écart de plus de 18 mm entre barycentres de masques, réglable par
`--group-gap-mm`).

```bash
./target/release/medx svr info --input stack_ax.nii.gz stack_cor.nii.gz --mask masque_ax.nii.gz masque_cor.nii.gz
```

## Limites

- **Un masque par stack est obligatoire** et sa qualité compte : le recalage et la reconstruction ne
  regardent que les pixels du masque. Le volume n'est donc défini que là où des pixels de masque
  l'atteignent (le cerveau et un peu autour) ; les coupes sans pixel de masque (hors du cerveau) ne
  contribuent pas, et les coupes de moins de 500 pixels de masque ne sont pas recalées (elles gardent la
  pose de leur en-tête).
- **Repère du volume** : le volume est ancré par les données. Avec `s2s`, rien ne fixe le repère (la solution
  est définie à un mouvement rigide commun près), donc la position et l'orientation du volume dans le repère du
  scanner ne sont pas garanties. Pour le comparer à un autre volume ou à un atlas, recalez-le rigidement.
- **Pas de correction d'intensité** : ni mise à l'échelle par coupe, ni champ de biais, ni
  pondération robuste des coupes aberrantes (coupes sombres). SVRTK le fait. Sur nos données TRUFI, les
  facteurs d'échelle mesurés sont petits, mais ce n'est pas démontré pour d'autres contrastes (HASTE :
  non évalué).
- **Fœtus très agités** : sur deux de nos six sujets, aucune méthode testée (la nôtre, SVRTK, NeSVoR)
  ne prédit mieux des coupes jamais vues que de ne pas corriger le mouvement.
- **Netteté** : avec α = 0,1, le hachurage régulier des premières versions a disparu, mais nos volumes sont plus lisses que ceux de SVRTK, qui préserve mieux les contours (lissage adaptatif). Baisser `--alpha` (0,03 à 0,05) rend l'image plus nette sur un sujet calme, au prix d'un peu de hachurage ; une régularisation qui préserve les contours est prévue.
- **Un seul jeu de données** (TRUFI, six sujets) ; pas de vérité terrain sur des acquisitions réelles.
- Seul un Mac avec CPU Apple a été testé. La commande est destinée à la **recherche**, pas à un usage
  médical ou diagnostique.

## Erreurs

La commande se termine par le code 0 si tout va bien, 1 en cas d'échec (message préfixé par
`medx svr recon :`), 2 si les arguments sont invalides.

| Message | Cause |
|---|---|
| `N masque(s) pour M stack(s) : il en faut un par stack` | `--mask` n'a pas autant de fichiers que `--input`. |
| `au moins deux stacks sont nécessaires pour estimer le mouvement` | Un seul stack : rien à croiser. |
| `… : masque incompatible avec le stack (…)` | Le masque n'est pas sur la grille du stack (dimensions ou affine). |
| `… : masque vide` | Le masque ne contient aucun voxel. |
| `… : pas de sform (affine absente)` | L'en-tête NIfTI n'a pas d'affine exploitable. |
| `… : écriture impossible : …` | Dossier de sortie absent ou non inscriptible. |
