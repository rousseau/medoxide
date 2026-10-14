# Algorithmes et références

## Fetal-BET

`medx fetalbet` reproduit l'inférence de **Fetal-BET**, un outil d'extraction
du cerveau fœtal en IRM fondé sur un Attention U-Net 2D. Le modèle
(architecture et poids) est celui des auteurs ; nous n'avons rien réentraîné.

> Faghihpirayesh R., Karimi D., Erdoğmuş D., Gholipour A. *Fetal-BET: Brain
> Extraction Tool for Fetal MRI*. IEEE Open Journal of Engineering in Medicine
> and Biology, 2024. Code : <https://github.com/IntelligentImaging/fetal-brain-extraction>

```bibtex
@article{faghihpirayesh2024fetal,
  title={Fetal-bet: Brain extraction tool for fetal mri},
  author={Faghihpirayesh, Razieh and Karimi, Davood and Erdo{\u{g}}mu{\c{s}}, Deniz and Gholipour, Ali},
  journal={IEEE Open Journal of Engineering in Medicine and Biology},
  year={2024},
  publisher={IEEE}
}
```

Si vous utilisez `medx fetalbet`, merci de citer cet article. Il est publié sous
licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), comme le
modèle. Les auteurs précisent que l'outil est destiné à la recherche, pas à un
usage médical ou diagnostique, et qu'il est fourni sans garantie.

## Pipeline

Pour chaque volume, `medx fetalbet` enchaîne :

1. **Rééchantillonnage à 1 × 1 mm** dans le plan (interpolation bilinéaire, zéros
   hors du volume). L'axe des coupes n'est pas modifié.
2. **Normalisation par coupe** : les voxels strictement positifs sont divisés par
   leur écart-type. La moyenne n'est *pas* soustraite : ce n'est pas un z-score.
3. **Inférence par fenêtres** de 256 × 256 voxels, avec 50 % de recouvrement :
   les sorties du modèle (logits) des fenêtres qui se recouvrent sont moyennées.
4. **Retour à la grille d'origine** : les logits sont rééchantillonnés vers la
   grille de l'image, puis on garde la classe de plus grand logit (cerveau ou
   fond). Interpoler le masque au lieu des logits donne un résultat différent.
5. **Écriture** du masque en NIfTI, avec l'en-tête de l'image d'entrée.

Ces étapes suivent le script d'inférence officiel de Fetal-BET. Le modèle est très
sensible à ce prétraitement : sur une coupe de test à 1,17 mm non rééchantillonnée,
il ne détectait aucun voxel de cerveau.

## Ce que nous avons modifié

Les poids PyTorch d'origine ont été exportés en ONNX, puis convertis en code Rust
et en poids au format `burnpack` pour [Burn](https://burn.dev) (backend `wgpu`,
donc exécution sur GPU). Les calculs sont équivalents à l'original
([Validation](validation.md)). Le prétraitement, les fenêtres et le
rééchantillonnage sont réécrits en Rust.

## Autres références

- **MONAI** fournit le modèle d'origine (`AttentionUnet`) et le script de référence
  (`Spacingd`, `SliceInferer`) auxquels nous comparons. Cardoso M. J. et al.,
  *MONAI: An open-source framework for deep learning in healthcare*,
  [arXiv:2211.02701](https://arxiv.org/abs/2211.02701), 2022.
- **Attention U-Net**, l'architecture. Oktay O. et al., *Attention U-Net: Learning
  Where to Look for the Pancreas*, [arXiv:1804.03999](https://arxiv.org/abs/1804.03999), 2018.
- **Burn**, le framework d'apprentissage profond en Rust : <https://burn.dev>.

## SVR, reconstruction coupe-vers-volume

`medx svr recon` reconstruit un volume 3D à partir de stacks de coupes 2D acquises avec du mouvement fœtal.
Le principe est celui de la reconstruction coupe-vers-volume (*slice-to-volume reconstruction*, SVR) de
Kuklisova-Murgasova et al. (2012) ; l'implémentation est la nôtre, en Rust et Burn, et fait des choix différents
de SVRTK et de NeSVoR (voir plus bas).

### Modèle d'acquisition

Chaque coupe est une version floue du volume, vue sous sa propre position : le volume est échantillonné
selon le repère de la coupe (affine de son en-tête, corrigée du mouvement estimé) puis lissé par une réponse
impulsionnelle (PSF) gaussienne orientée avec la coupe : écart type dans le plan = 1,2 × taille du pixel / 2,355,
hors plan = épaisseur / 2,355. L'**opérateur adjoint** (la rétroprojection) est écrit et vérifié exactement.
Seuls les pixels du **masque cérébral** de la coupe sont utilisés.

### Reconstruction du volume

Le volume est discrétisé sur une grille isotrope (0,8 mm par défaut) autour des masques, avec une marge de
10 mm. Pour des poses données, on minimise

> ½ Σ ‖ m · (A x − y) ‖² + (α / 2) ‖ G x ‖²

où `A` est l'opérateur d'acquisition de chaque coupe, `y` ses pixels, `m` son masque et `G` le gradient du
volume (régularisation de Tikhonov d'ordre 1, `α` = 0,1 par défaut, voir [Validation](validation.md#choix-de-la-régularisation-α)). La solution est calculée par
**gradient conjugué** sur les équations normales, à partir de l'adjoint normalisé. Nous avons comparé six
optimiseurs sur cet objectif (gradient conjugué, plus forte pente, Barzilai-Borwein, Jacobi, Adam et L-BFGS
de Burn) : le gradient conjugué est le plus précis.

### Estimation du mouvement

Chaque coupe a un mouvement **rigide** propre (3 rotations, 3 translations), exprimé autour du barycentre du
masque ; les paramètres sont estimés par **Adam**, avec le gradient calculé par la **différentiation
automatique de Burn**. Deux méthodes, au choix (`--motion`) :

- **Coupe-coupe (`s2s`)** : deux coupes de stacks différents se croisent le long d'une droite, où leurs
  intensités doivent concorder. Le coût d'une coupe est la corrélation normalisée moyenne des profils
  d'intensité échantillonnés (tous les 0,5 mm) sur ses droites d'intersection avec les coupes des autres
  stacks. Quand la pose change, la droite bouge : le coût est dérivable à travers la géométrie. Toutes les
  coupes sont recalées l'une après l'autre en **balayages** (ordre aléatoire à graine fixe, chaque coupe voit
  les poses déjà mises à jour), une mise à jour n'étant acceptée que si la concordance ne baisse pas. Aucun
  volume n'est reconstruit pendant l'estimation ; le volume est reconstruit une fois à la fin.
- **Coupe-volume (`s2v`)** : chaque coupe est recalée contre un volume reconstruit, par corrélation
  normalisée. La référence **ne contient pas la coupe recalée** (sa contribution est retirée exactement de
  l'adjoint normalisé) : sinon chaque coupe retrouverait sa propre erreur dans la référence. Les coupes
  mal recalées (corrélation finale < 0,9) sont relancées depuis 6 poses de départ tirées dans ±5 mm. Les
  coupes sont recalées en 6 cycles (chaque cycle repart des poses du précédent) ; la référence ne dépendant
  pas du volume reconstruit, celui-ci n'est calculé qu'une fois, à la fin.

**Alignement par stack ou par paquets (optionnel, `--s2s-block-levels`).** Les coupes d'un stack sont acquises l'une après l'autre (environ 1 s par coupe en TRUFI) : un fœtus qui bouge entre deux stacks déplace d'un bloc toutes les coupes
d'un stack, ce que le recalage coupe par coupe ne rattrape pas au-delà d'environ 4 mm. Avant les balayages par coupe, chaque **bloc de coupes consécutives** reçoit une transformation rigide **commune** (6 paramètres autour du
barycentre des pivots du bloc), estimée en maximisant la concordance moyenne du bloc : une recherche exhaustive (translations d'une grille de ±12 mm au pas de 4 mm, puis rotations de la même grille en mm équivalents), puis Adam. La
transformation n'est acceptée que si la concordance du bloc ne baisse pas. Niveau 0 : le stack entier (le premier stack sert d'ancre) ; niveau *l* : 2^*l* blocs par stack, du gros vers le fin ; chaque niveau a `--s2s-block-sweeps`
balayages (les stacks se voient entre eux avec leurs poses déjà corrigées). Il joue le rôle du recalage volume-vers-volume des stacks de NiftyMIC et de l'alignement global de SVRTK, mais avec le même critère que le recalage par coupe.

### Ce que SVRTK fait et que nous ne faisons pas (encore)

SVRTK, la référence de la communauté, ajoute notamment l'alignement global des stacks, la mise à l'échelle
des intensités par coupe, un champ de biais, des statistiques robustes avec exclusion de coupes et un lissage
adaptatif préservant les contours. Nous n'avons rien de tout cela dans la chaîne par défaut (une mise à
l'échelle par coupe est écrite et testée mais non branchée). À l'inverse, notre opérateur a un adjoint exact,
notre objectif est écrit (donc validable), et les références « sans la coupe » n'existent pas chez SVRTK, qui
recale contre un volume contenant la coupe.

### Références

- **Principe de la SVR fœtale** : Kuklisova-Murgasova M., Quaghebeur G., Rutherford M. A., Hajnal J. V.,
  Schnabel J. A. *Reconstruction of fetal brain MRI with intensity matching and complete outlier removal*.
  Medical Image Analysis 16(8):1550-1564, 2012. <https://doi.org/10.1016/j.media.2012.07.004>
- **SVRTK**, comparé à nos résultats (utilisé comme outil externe, licence Apache 2.0) :
  <https://github.com/SVRTK/SVRTK>. Référence demandée par le projet : Uus A. et al. *Deformable
  Slice-to-Volume Registration for Motion Correction in Fetal Body MRI and Placenta*. IEEE Transactions on
  Medical Imaging 39(9):2750-2759, 2020. <https://doi.org/10.1109/TMI.2020.2974844>
- **NeSVoR**, comparé à nos résultats (modèle d'acquisition et estimation du mouvement coupe-volume sans
  SVoRT) : Xu J., Moyer D., Gagoski B., Iglesias J. E., Grant P. E., Golland P., Adalsteinsson E. *NeSVoR:
  Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI*. IEEE Transactions on Medical
  Imaging, 2023. <https://doi.org/10.1109/TMI.2023.3236216>
- **NiftyMIC**, comparé à nos résultats (utilisé comme outil externe, image Docker `renbem/niftymic`, licence BSD-3 ; recalage volume-vers-volume des stacks, puis coupe-volume hiérarchique, correction d'intensité et rejet des
  coupes de corrélation < 0,5 puis 0,65 puis 0,8) : Ebner M., Wang G., Li W., Aertsen M., Patel P. A., Aughwane R., Melbourne A., Doel T., Dymarkowski S., De Coppi P., David A. L., Deprest J., Ourselin S., Vercauteren T.
  *An automated framework for localization, segmentation and super-resolution reconstruction of fetal brain MRI*. NeuroImage 206:116324, 2020. <https://doi.org/10.1016/j.neuroimage.2019.116324>
- **Atlas fœtal** utilisé pour les tests sur données simulées : Gholipour A. et al. *A normative
  spatiotemporal MRI atlas of the fetal brain for automatic segmentation and analysis of early brain
  growth*. Scientific Reports 7:476, 2017. <https://doi.org/10.1038/s41598-017-00525-w>
