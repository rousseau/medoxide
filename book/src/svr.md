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
| `--s2s-highpass-mm` | Avec `s2s` : passe-haut des profils d'intensité comparés (écart type, en mm, du lissage soustrait) pour ignorer les variations lentes d'intensité entre stacks (champ de biais, gain). Sans lui, `s2s` s'effondre en simulation avec un biais de 30 % ; 0 : aucun. | 25 |
| `--s2s-coarse-sigma`, `--s2s-coarse-sweeps`, `--s2s-coarse-levels` | Balayages préliminaires du gros vers le fin (coupes floutées de σ pixels, flou divisé par 2 à chaque niveau) qui élargissent la portée de capture. **Défaut σ = 1, 3 balayages, 2 niveaux** : sur données réelles, avec une référence indépendante et trois graines, la part des coupes retrouvées passe de 29 à 66 % pour une perturbation indépendante de ±3 mm. Un flou plus fort (σ = 6, réglé sur la simulation) **réduit** la portée réelle et dégrade la concordance : à éviter. 0 pour σ : désactivé. Voir [Validation](validation.md#un-flou-adapté-aux-données-réelles-la-portée-de-capture-mesurée-avec-une-référence-de-consensus). | 1, 3, 2 |
| `--min-stack-coverage` | Nombre minimal de stacks qui doivent atteindre un voxel pour qu'il soit reconstruit. Avec 2 (défaut), les zones vues par un seul stack, souvent des coupes mal placées qui forment des « languettes » hors du cerveau, ne sont pas reconstruites ; 1 : tous les voxels touchés. | 2 |
| `--edge-kappa` | **Expérimental.** Reconstruction préservant les contours (Huber, trois repondérations) : seuil relatif κ, un contour étant une différence entre voxels voisins supérieure à κ fois la différence médiane ; 0 : lissage de Tikhonov seul. Sur données réelles, aucun gain démontré (voir [Validation](validation.md#régularisation-préservant-les-contours)). | 0 |
| `--quality` | Écrit un fichier d'indicateurs par coupe ([format](#fichier-de-qualité)). | aucun |
| `--s2s-block-levels`, `--s2s-block-sweeps`, `--s2s-block-search-mm` | Alignement par stack ou par paquets avant les balayages par coupe (niveau 0 : stack entier ; niveau *l* : 2^*l* blocs de coupes consécutives ; recherche exhaustive de ±`search` mm puis Adam). Corrige les décalages de stack de 4 à 10 mm en simulation et rapproche `s2s` de NiftyMIC et SVRTK sur un sujet agité réel, sans régression sur trois sujets calmes ; coûte environ 1,5 fois le temps de calcul ; 0 niveau : désactivé. Sans effet sur le sujet très agité testé ; 3 niveaux n'aident que pour un décalage de stack **avec** biais d'intensité. Voir [Validation](validation.md#alignement-par-stack-ou-par-paquets). | 1, 2, 12 |
| `--s2s-bias-rounds`, `--s2s-bias-min-reduction` | Correction du champ de biais par stack : nombre d'alternances « estimer le champ de chaque stack (polynôme de degré 3, sans volume), diviser les stacks, recaler » ; le volume est reconstruit à partir des stacks corrigés. La correction n'est appliquée que si elle réduit d'au moins `min-reduction` (10 %) l'incohérence d'intensité entre stacks : sans biais, rien ne change. En simulation avec un biais de 30 %, de 3 à 75 % (mouvement lisse) et de 71 à 79 % (indépendant) des coupes à ≤ 1,5 mm avec l'alignement par stack ; sur données réelles, concordance inchangée à 0,01 près sur 6 sujets et sujet très agité remis au consensus de NiftyMIC et NeSVoR. 0 alternance : désactivé. Voir [Validation](validation.md#correction-du-champ-de-biais-par-stack). | 1, 0,1 |
| `--s2s-temporal-weight` | Régularisation temporelle : les coupes d'un stack sont acquises l'une après l'autre (≈ 1 s), donc deux voisines de stack ont presque le même mouvement. Le coût d'une coupe est la concordance moins λ fois l'écart quadratique moyen (mm²) entre ses points déplacés par sa pose et par le mouvement prédit par ses deux voisines directes. Réglé à 0,01 par le plus petit λ satisfaisant cinq critères fixés avant la mesure (voir [Validation](validation.md#régularisation-temporelle)) : sur un sujet calme (banc) de 23 à 85 % (mouvement lisse ±6 mm) et de 66 à 90 % (indépendant ±3 mm) des coupes à ≤ 1,5 mm ; concordance réelle à 0,002 près, temps CPU +5 % en moyenne. 0 : désactivé. | 0,01 |
| `--s2s-temporal-robust-mm`, `--s2s-temporal-robust-start-mm` | Pénalité temporelle **robuste** de Geman-McClure (`d²/(1 + d²/c²)` par point, `c` en mm) au lieu de la quadratique ; avec un départ `c0 > 0`, l'échelle décroît géométriquement de `c0` à `c` au fil des balayages. Désactivées par défaut (0) : sur le banc semi-synthétique elles corrigent le cas d'un mouvement indépendant d'une coupe à l'autre (de 42 % à 85 % des coupes à ≤ 1,5 mm avec `c0 = 8`, `c = 2`), mais sur le banc de portée réel (mouvement lisse de ±6 mm) elles tombent de 85 % à 45 % : un petit `c` final ne corrige plus les grands décalages. Voir [Validation](validation.md#régularisation-temporelle). | 0, 0 |
| `--s2s-temporal-info-exponent` | Pondère le poids temporel de chaque coupe par son information : `λ_k = λ · clamp((N_médian / N_k)^γ, ¼, 4)`, `N_k` étant son nombre de points d'intersection valides (une coupe de bord, peu contrainte, serait tirée plus fort vers ses voisines). Désactivé par défaut (0) : sur le banc semi-synthétique, γ = 0,5 et γ = 1 ne gagnent rien en lisse ±6 mm (31 et 26 % contre 29 %) et perdent jusqu'à 9 points en sauts. Voir [Validation](validation.md#régularisation-temporelle). | 0 |
| `--s2v-refine-cycles`, `--s2v-refine-resolution-mm` | Raffinement coupe-volume après le coupe-coupe : `N` cycles de la boucle coupe-volume partant des poses du coupe-coupe, sur les stacks corrigés du biais, avec une reconstruction à la résolution indiquée (2 mm, pour borner le coût) ; la reconstruction finale se fait ensuite à la résolution demandée. **Désactivé par défaut (0)** : sur le banc semi-synthétique il gagne +27 à +48 points (mouvements difficiles) sans dégrader la médiane, mais sur six sujets réels l'accord des poses avec SVRTK, NeSVoR et NiftyMIC baisse légèrement (part de coupes à plus de 3 mm de 20,6 à 24,2 % en moyenne) et la concordance de 0,813 à 0,773. Voir [Validation](validation.md#régularisation-temporelle). | 0, 2 |
| `--s2s-exclude-fraction` | Écarte de la RECONSTRUCTION la fraction indiquée (0 à 1) des coupes recalées de plus faible concordance finale (masque de coupe vidé ; les poses ne changent pas). **Désactivé par défaut (0)** : la concordance finale prédit modérément l'échec de pose (AUC 0,72 à 0,86 sur le banc semi-synthétique), mais exclure 15 % des coupes dégrade la netteté de 2 à 4 % et le juge hors échantillon sur trois sujets. Voir [Validation](validation.md#régularisation-temporelle). | 0 |
| `--s2s-sweeps` | Avec `s2s` : nombre de balayages de toutes les coupes. | 3 |
| `--s2s-update` | Avec `s2s` : mise à jour des poses pendant un balayage. `gauss-seidel` : les coupes sont recalées l'une après l'autre, chacune voit les poses déjà mises à jour (séquentiel). `jacobi` : toutes les coupes d'un balayage sont recalées en parallèle contre les poses du balayage précédent. Voir [ci-dessous](#gauss-seidel-ou-jacobi). | `gauss-seidel` |
| `--s2v-cycles` | Avec `s2v` : nombre de cycles « recaler puis reconstruire ». | 6 |

## Dans une arborescence BIDS

Avec `--bids`, la commande lit et écrit dans les dossiers que lit [guioxide](https://github.com/rousseau/guioxide),
sans nommer un fichier. Les piles sont les `*_T2w.nii.gz` de `sub-X[/ses-Y]/anat/` ; leurs masques sont ceux de
`medx fetalbet --bids` (`derivatives/medx-fetalbet/…/<entités de la pile>_desc-brain_mask.nii.gz`).

```bash
./target/release/medx fetalbet --bids /chemin/jeu --participant 01 --session 1
./target/release/medx svr recon --bids /chemin/jeu --participant 01 --session 1
```

`<étude>` désigne la racine du jeu, ou son parent si elle s'appelle `sourcedata` (buckets Garage : `sourcedata/` et
`derivatives/` côte à côte) ; `--output-dir <dossier>` la remplace, pour les masques comme pour les résultats.
`--input` restreint la liste des piles (au moins deux). Les résultats vont dans
`<étude>/derivatives/medx-svr/sub-X[/ses-Y]/anat/` :

| Fichier | Contenu |
|---|---|
| `sub-X_ses-Y_desc-svr_T2w.nii.gz` | volume reconstruit (`sform` et `qform` écrits) |
| `sub-X_ses-Y_desc-svr_T2w.json` | résolution, mode de mouvement, réglages, piles utilisées (noms sans extension), ligne de commande |
| `sub-X_ses-Y_desc-svr_motion.tsv` | mouvement estimé de chaque coupe |
| `sub-X_ses-Y_desc-svr_slicequality.tsv` | qualité de chaque coupe |

et `<étude>/derivatives/medx-svr/dataset_description.json` (`DatasetType: derivative`, `GeneratedBy` : nom, version,
commit, ligne de commande). `--output`, `--mask`, `--poses` et `--quality` servent hors BIDS et sont refusés avec `--bids`.

L'affine d'un stack (et de son masque) suit la règle de nibabel et de guioxide : le `sform` s'il est défini, sinon le `qform`, sinon la diagonale des `pixdim` (volume centré sur l'origine). Un stack qui n'a qu'un `qform` est donc lu, et non refusé.

## Choisir l'estimation du mouvement

| `--motion` | Principe | Durée | À retenir |
|---|---|---|---|
| `s2s` (défaut) | Chaque coupe est recalée contre les coupes des **autres stacks** qui la croisent, sans reconstruire de volume intermédiaire. | environ 4 minutes pour 3 stacks (98 coupes), sur un Mac à 12 cœurs (2,5 minutes sans l'alignement par stack) | Rapide. Sur nos six sujets, la qualité de la reconstruction est comparable à celle de `s2v`. |
| `s2v` | Chaque coupe est recalée contre une référence calculée à partir des **autres** coupes (elle ne contient pas la coupe recalée), en 6 cycles ; le volume n'est reconstruit qu'à la fin, avec les poses finales. | environ 2,5 minutes pour 3 stacks (98 coupes) sur un Mac à 12 cœurs | Plus lent. Méthode classique de l'état de l'art (coupe-volume). |

Les deux méthodes produisent les mêmes sorties (une transformation rigide par coupe),
et la reconstruction du volume est la même.

## Gauss–Seidel ou Jacobi

Le recalage coupe-coupe fait plusieurs balayages de toutes les coupes. Par défaut (`--s2s-update gauss-seidel`) il les recale **une à une** : la phase est séquentielle (un seul cœur) et ses résultats sont ceux de toutes les mesures de cette page. `--s2s-update jacobi` recale toutes les coupes d'un balayage **en parallèle**, contre les poses figées au début du balayage, puis les met à jour ensemble. Le résultat ne dépend ni de l'ordre des coupes ni du nombre de threads (poses identiques au bit près avec 1, 3 ou 8 threads).

Mesures (3 stacks, 6 balayages nets, un lancement par cas, Mac à 12 cœurs partagé avec d'autres calculs ; `scripts/analyse_jacobi.py`) :

| Jeu | Durée totale GS / Jacobi | Par balayage net | Erreur médiane de pose | Concordance finale |
|---|---|---|---|---|
| Réel (un sujet, trois stacks trufi) | 913 / 479 s | 48 / 9 s | (pas de vérité) | 0,866 / 0,867 |
| Simulé, mouvement lisse de 4 mm | 960 / 471 s | 50 / 9 s | 0,46 / 0,48 mm | 0,914 / 0,915 |
| Simulé, mouvement lisse de 8 mm | 974 / 599 s | 51 / 15 s | 3,75 / 4,32 mm | 0,668 / 0,617 |
| Simulé, mouvement non lisse de 8 mm | 839 / 455 s | 40 / 9 s | 9,19 / 10,09 mm | 0,268 / 0,249 |

Pour un mouvement modéré, Jacobi donne la même qualité en deux fois moins de temps (le balayage est 4 à 5 fois plus rapide, mais l'alignement par blocs, séquentiel, reste). Il consomme environ 1,4 fois plus de CPU au total. Pour un mouvement de 8 mm, **aucune des deux variantes ne corrige correctement** (l'erreur dépasse même celle sans correction pour le mouvement non lisse), et Jacobi est un peu moins bon. Il converge aussi plus lentement (déplacement moyen sous 0,1 mm au balayage 2 contre 1 sur le sujet réel). Il n'a pas été validé sur les six sujets réels : c'est pourquoi Gauss–Seidel reste le défaut.

Profil par phase (3 stacks réels trufi, réglages par défaut, un Mac à 12 cœurs ; `MEDX_S2S_TRACE=1` affiche la durée de chaque phase) :

| Phase | Avant la recherche parallèle (Gauss–Seidel ; phases estimées par des mesures isolées) | Gauss–Seidel | Jacobi |
|---|---|---|---|
| Alignement par blocs | ≈ 340 s | 140 s | 134 s |
| 9 balayages (6 grossiers, 3 nets) | ≈ 420 s | 432 s (48 s chacun) | 80 s (9 s chacun) |
| Champ de biais (2 fois) | 2 s | 2 s | 2 s |
| Reconstruction du volume | 25 s | 25 s | 25 s |
| **Total mesuré** (horloge du programme) | **775 s** | **603 s** | **242 s** |

La recherche exhaustive de l'alignement par blocs (686 évaluations indépendantes par bloc) est évaluée en parallèle ; les poses de Gauss–Seidel sont identiques au bit près à celles d'avant (comparées sur ce calcul complet). Avec Jacobi, l'alignement par blocs pèse 55 % du temps, les balayages 33 % et la reconstruction 10 % ; le CPU moyen est de 5,4 cœurs sur 12. Le temps affiché par `/usr/bin/time` inclut une éventuelle mise en veille de la machine, pas celui du programme (un lancement a affiché 1326 s pour 603 s de calcul).

`MEDX_S2S_TRACE=1` affiche aussi une ligne par balayage (nombre de mises à jour, concordance, déplacement moyen, durée) et la concordance finale mesurée de la même façon pour les deux variantes ; `MEDX_S2S_TRACE_DIR=<dossier>` écrit les poses après chaque balayage.

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

Avec `--poses` (ou `_desc-svr_motion.tsv` en BIDS), un fichier TSV est écrit : une ligne d'en-tête, puis **une ligne par coupe**, terminé par un saut de ligne.

```text
stack<TAB>slice<TAB>m00<TAB>m10<TAB>m20<TAB>m30<TAB>m01<TAB>m11 … m33
```

`stack` est le **nom du fichier du stack sans extension** (`sub-X_ses-Y_acq-ax_run-1_T2w`) : deux stacks de même nom sont refusés. `slice` est l'indice de la coupe dans son stack (0 = première).
Les 16 valeurs sont la matrice `M` de 4 × 4 **en ordre colonne** (`m<ligne><colonne>`) : le mouvement rigide de la coupe, en coordonnées monde (mm, RAS+),
**absolu** par rapport à la géométrie d'en-tête de son stack. La coupe `k` d'un stack d'affine d'en-tête `A` est placée par `M · A · T(0, 0, k)`, où `T(0, 0, k)` est la translation de `k` voxels le long de la troisième
colonne de `A` : `M` est composé **après** l'affine de l'en-tête. En BIDS, ce sens est aussi écrit dans `_desc-svr_T2w.json` (clé `MotionConvention`). En Python :

```python
m = np.array([float(v) for v in champs[2:]]).reshape(4, 4, order="F")
```

Les nombres sont écrits avec assez de chiffres pour être relus exactement. Avant cette version, le fichier n'avait pas d'en-tête et `stack` était la position du stack dans `--input` ; les scripts de `scripts/` lisent les deux formes.

## Fichier de qualité

Avec `--quality` (ou `_desc-svr_slicequality.tsv` en BIDS), un fichier TSV est écrit : une ligne d'en-tête, puis **une ligne par coupe**, terminé par un saut de ligne. En-tête : `stack slice registered score_before score_after n_registrations correction_rms_mm excluded reason`.

| Colonne | Sens |
|---|---|
| `stack` | nom du fichier du stack sans extension (comme dans le fichier de poses) |
| `slice` | indice de la coupe dans son stack (0 = première) |
| `registered` | 1 si la coupe a été recalée ; 0 si elle a moins de 500 pixels de masque (elle garde sa pose) |
| `score_before`, `score_after` | avec `s2s`, concordance des profils d'intersection avant et après le dernier balayage ; avec `s2v`, corrélation avec la référence ; de −1 à 1, plus grand = mieux. `n/a` pour une coupe non recalée |
| `n_registrations` | nombre de recalages |
| `correction_rms_mm` | déplacement quadratique moyen des pixels de masque entre la pose de départ et la pose finale |
| `excluded` | 1 si la coupe est retirée de la reconstruction, sinon 0 |
| `reason` | motif d'exclusion, `n/a` sinon : `faible_concordance` (écartée par `--s2s-exclude-fraction`) ou `hors_masque` (aucun pixel de masque : elle ne contribue pas) |

Une valeur absente s'écrit `n/a`. Ces indicateurs, étudiés seulement avec `s2s`, permettent de reconnaître le **régime** (peu ou presque toutes les coupes mal recalées), pas de compter les coupes mal recalées de façon fiable ([Validation](validation.md#combien-de-coupes-sont-mal-recalées)). Avant cette version, l'en-tête était `stack coupe recalee score_avant …`, `stack` était un indice et une coupe non recalée avait des scores nuls.

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
- **Correction d'intensité partielle** : un champ de biais polynomial par stack est corrigé (voir `--s2s-bias-rounds`), mais ni la mise à l'échelle par coupe, ni la
  pondération robuste des coupes aberrantes (coupes sombres). SVRTK le fait. Sur nos données TRUFI, les
  facteurs d'échelle mesurés sont petits, mais ce n'est pas démontré pour d'autres contrastes (HASTE :
  non évalué).
- **Fœtus très agités** : sur deux de nos six sujets, aucune méthode testée (la nôtre, SVRTK, NeSVoR)
  ne prédit mieux des coupes jamais vues que de ne pas corriger le mouvement avec le protocole à deux stacks. Les méthodes y
  divergent entre elles de 6 à 13 mm : le mouvement dépasse la portée de capture des recalages locaux (environ ±4 mm en simulation),
  et l'élargir demande de traiter d'abord les incohérences d'intensité entre stacks ([explication](validation.md#pourquoi-certains-sujets-très-agités-ne-sont-pas-corrigés)).
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
| `… : écriture impossible : …` | Dossier de sortie absent ou non inscriptible. |
