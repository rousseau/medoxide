# Recherche : optimisation multi-départs pour le recalage coupe-coupe (2026-10-15)

Objectif : élargir la portée de capture réelle du recalage par coupe (mesurée à ≈ ±2 mm avec le recalage fin seul, ±3 mm avec le flou σ = 1, 0 % à ±6 mm) sans dépendre d'un volume reconstruit.

## 1. FLIRT (Jenkinson & Smith 2001, *Med. Image Anal.* 5(2):143-156 ; Jenkinson et al. 2002, *NeuroImage* 17(2):825-841)

Source lue : rapport technique FMRIB TR00MJ2 (sections « Global Optimisation Method »), documentation du langage de plannings.

- **Constat de départ** : l'optimisation locale + pyramide multirésolution « n'est pas suffisante pour trouver de façon fiable le minimum global ».
- **Pyramide 8 / 4 / 2 / 1 mm**, avec des **pas liés à la résolution** (déplacement maximal d'un voxel sous la moitié d'un voxel : translation ≤ n/2, rotation ≤ n/(2R), R = rayon du cerveau) et une terminaison de recherche linéaire (Brent) dès que le minimum est encadré à moins d'un pas.
- **Recherche initiale à 8 mm** : grille grossière de M³ angles d'Euler ; **à chaque point de la grille, optimisation locale des translations et de l'échelle** (4 degrés de liberté, depuis t = 0, s = 1) ; puis grille fine de N³ angles (N > M ; valeurs réduites M = 6, N = 20 pour le temps de calcul) où la translation et l'échelle sont **interpolées** des voisins grossiers et où l'on **évalue seulement le coût** ; minima locaux de la grille fine ; optimisation complète (7 ddl) depuis chacun.
- **Ré-optimisation avec perturbations à 4 mm** : les 3 meilleures solutions de 8 mm (avant et après optimisation) et **10 perturbations** de chacune (± ½ Δθ sur chacun des 3 angles, ± Δs et ± 2 Δs sur l'échelle) : 33 optimisations ; étape jugée « critique pour éliminer de nombreux mauvais recalages ».
- **Raffinement** à 2 mm (7, 9 puis 12 ddl), puis une passe à 1 mm.
- **Idée à retenir** : rechercher exhaustivement la **rotation** (la dimension non convexe) et résoudre la **translation localement** à chaque candidat ; garder plusieurs candidats d'un niveau à l'autre ; perturber pour s'échapper de bassins voisins.

## 2. ROSI / pyrecon (Mercier et al., *Computers in Biology and Medicine*, 2025 ; thèse C. Mercier, IMT Atlantique)

Source lue : dépôt `pyrecon` (clone local), fichier `ROSI/rosi/registration/outliers_detection/multi_start.py`.

1. Un **classifieur** (forêt aléatoire) repère les coupes mal recalées (caractéristiques tirées des coûts d'intersection, Dice…) ; le multi-départ ne s'applique qu'à elles.
2. **Départs issus des voisines** : pour une coupe suspecte, les voisines bien recalées du même stack (3 avant, 3 après, de 2 en 2) fournissent des poses de départ : **moyenne géodésique de deux rotations** et moyenne des translations (paires avant × après), ou extrapolation depuis une seule voisine ; jusqu'à 15 candidats, dédoublonnés (rotations à moins de 0,2 rad = 11°).
3. **Grille de rotations 5 × 5 × 5** autour de chaque départ (± 3° et ± 6° sur chaque angle) avec, à chaque point, **optimisation des translations seules** (Nelder–Mead) : c'est la recherche de FLIRT, à l'échelle d'une coupe.
4. **Minima locaux** de la grille (6-voisinage) ; on garde les **5 meilleurs** ; optimisation à 6 paramètres depuis chacun ; on garde le meilleur **seulement s'il fait mieux que le coût initial**.
5. Réitération avec reclassement ; coût combiné (concordance + Dice des recouvrements, poids ω).

## 3. Autres pistes d'optimisation (non retenues pour l'instant)

- Initialisation apprise (SVoRT dans NeSVoR : transformeur prédisant les poses) : rapide mais dépend d'un entraînement ; hors périmètre « implémentation commune sans dépendance ».
- Recherche globale à base de recalage volume-à-volume des stacks avec recherche angulaire de 45° (NiftyMIC) : traitée par l'alignement par stack.
- Optimiseurs globaux génériques (CMA-ES, recuit, branch-and-bound de type Go-ICP, recherche de rotation dans l'espace de Fourier) : coût élevé pour six paramètres par coupe ; à reconsidérer si la grille + raffinement échoue.

## 4. Ce que cela suggère pour medoxide

Deux idées **distinctes**, à évaluer séparément :
- **(a) Grille de rotations avec translation ré-optimisée** (FLIRT, ROSI) : robustesse à l'initialisation, **sans hypothèse sur les voisines**.
- **(b) Départs issus des voisines** (ROSI) : continuité temporelle (les coupes d'un stack sont acquises l'une après l'autre) ; utile quand les voisines sont bien recalées et que le mouvement est lisse ; inutile si le mouvement est indépendant d'une coupe à l'autre.

Banc d'essai : le diagnostic `diagnose_the_capture_range_on_real_data` (référence de consensus, perturbation connue) a maintenant deux modes de perturbation : `independante` (teste (a)) et `lisse` (marche autorégressive, ρ = 0,9 ; teste (a) et (b)).

Étapes (une à la fois, pause après chacune) :
1. **Mesure de base** sur le banc d'essai : recalage fin, flou σ = 1 (défaut), amplitudes 3, 4, 6 mm, perturbation indépendante et lisse.
2. **Candidats et sélection sans optimisation** : générer les départs (pose courante, grille de rotations autour de chacun, éventuellement poses interpolées des voisines) ; évaluer le coût de concordance (passe avant, sans gradient) ; mesurer, sur le banc, la part des coupes dont le meilleur candidat est à ≤ 1,5 mm de la référence (« le bon bassin est-il dans la liste ? »).
3. **Raffinement** des K meilleurs minima locaux (Adam à 6 paramètres) et garde « accepter seulement si la concordance augmente ».
4. **Déclencheur** : n'appliquer le multi-départ qu'aux coupes suspectes (concordance sous un seuil relatif à leur stack) ; sinon le coût est multiplié.
5. **Évaluation** : banc d'essai réel ; pipeline depuis l'en-tête sur les six sujets (concordance et accord des poses) ; non-régression en simulation ; temps de calcul.

**Mesures de base (étape 1, calme 2, part des coupes retrouvées à ≤ 1,5 mm)** :

| Amplitude | indépendante, fin seul | indépendante, flou σ = 1 (défaut) | lisse, fin seul | lisse, flou σ = 1 (défaut) |
|---|---|---|---|---|
| ±0 | 100 % | 97 % (σ = 6) | – | – |
| ±3 mm / ° | 11 % | 72 % | 89 % | 86 % |
| ±4 mm / ° | 1 % | 37 % | 52 % | 77 % |
| ±6 mm / ° | 0 % | 0 % | 0 % | 10 % |

Le régime lisse est nettement plus facile : les erreurs de départ des voisines sont corrélées, donc les paires de coupes restent presque cohérentes. Le régime indépendant est le cas difficile, où seule la grille de rotations (a) peut aider (les voisines sont aussi fausses).

Critères proposés (à valider avant de coder) :
- banc réel, perturbation indépendante, **±4 mm : ≥ 70 %** des coupes à ≤ 1,5 mm (37 % avec le flou seul) ; **±6 mm : ≥ 30 %** (0 %) ; perturbation lisse, **±6 mm : ≥ 50 %** (10 %) ;
- départ à la référence (±0) : ≥ 97 % conservées ;
- pipeline depuis l'en-tête sur six sujets réels : aucune baisse de concordance supérieure à 0,01, moyenne en hausse d'au moins 0,01 ; accord des poses avec NiftyMIC / SVRTK / NeSVoR non dégradé sur le sujet calme ; simulation ±4 mm sans régression ;
- coût en temps ≤ 1,5 fois celui du défaut.
