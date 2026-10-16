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

## 5. Résultat de l'étape 2 : le bon bassin est parfois dans la liste, mais on ne sait pas le choisir (2026-10-15)

Diagnostic `diagnose_the_multistart_candidates_on_real_data` (calme 2, référence de consensus ; une coupe sur 2 ou 3 de chaque stack ; grille de rotations ±3°/±6° (5³ = 125 candidats), translation ré-optimisée à chaque point (Adam, 20 itérations), images floutées de σ = 1 ; part des coupes à ≤ 1,5 mm de la référence) :

| Situation | coupes | erreur de départ (médiane) | Adam depuis le départ | meilleur candidat (concordance) | meilleur des 5 maxima | un des 125 candidats |
|---|---|---|---|---|---|---|
| ±4 mm indépendant, partenaires perturbés (1er balayage) | 10 | 3,6 mm | 0 % | 0 % | 0 % | 30 % |
| ±4 mm indépendant, partenaires à la vraie pose (borne haute) | 10 | 3,6 mm | 90 % | 90 % | 90 % | 100 % |
| ±4 mm indépendant, **après les balayages actuels** (coupes restées > 1,5 mm) | 33 | 3,1 mm | 3 % | 3 % | 3 % | 39 % |
| même, partenaires restreints aux coupes bien recalées (oracle) | 28 | 3,1 mm | 4 % | 4 % | 25 % | 46 % |
| ±6 mm lisse, après les balayages actuels | 32 | 6,4 mm | 0 % | 0 % | 0 % | 3 % |

Lecture :
- **Avec des partenaires justes, le recalage actuel suffit déjà à ±4 mm (90 %)** : la grille n'apporte rien. Le problème n'est pas la capture d'une coupe isolée mais la **cohérence conjointe** : les partenaires sont eux aussi faux.
- **Après les balayages**, le bon bassin est dans la grille pour 39 % des coupes mal recalées (±4 mm), mais le classement par concordance ne le retrouve que dans 3 % des cas (5 maxima : 3 %) : c'est un problème de **sélection**, pas de génération. Restreindre les partenaires aux coupes bien recalées (oracle) l'améliore (25 % avec les 5 maxima) sans le résoudre : il ne reste que 37 % de partenaires corrects, donc peu de droites d'intersection fiables par coupe.
- **À ±6 mm**, la grille (±6°, translations par 20 itérations d'Adam) ne contient presque jamais le bon bassin (3 %), et il ne reste presque aucune coupe bien recalée pour servir de partenaire.
- Le critère de départ (±4 mm indépendant ≥ 70 %) n'est donc **pas atteignable par un multi-départ par coupe isolé** dans ce régime : l'étape 3 (raffiner les K meilleurs) n'a pas de sens si la sélection échoue. Le banc d'essai indépendant est un régime sévère (chaque coupe est fausse indépendamment des voisines) : le mouvement réel est plutôt lisse.

Pistes pour la suite (à choisir) :
- (A) **Pondérer les partenaires par leur fiabilité** (cœur de ROSI : le coût n'utilise que les coupes jugées bonnes, via un classifieur) plutôt que de tout croiser avec tout ; estimation de la fiabilité par la concordance de chaque coupe, réestimée à chaque balayage.
- (B) **Multi-départ au niveau du bloc ou du stack** (peu d'inconnues, partenaires nombreux) avec grille de rotations comme FLIRT, puis par coupe : l'alignement par stack actuel ne cherche que des translations exhaustivement, pas les rotations.
- (C) **Régularisation temporelle** : pénaliser l'écart de pose entre coupes voisines d'un stack (équivalent doux des départs issus des voisines de ROSI) ; n'aide que si le mouvement est lisse, donc à évaluer avec la perturbation `lisse`.

## 6. Piste A (pondérer les partenaires par leur fiabilité) : rejetée (2026-10-15)

Implémentation : `IntersectionProblem::with_partner_weights` (moyenne pondérée des corrélations, testée exactement : poids égaux = moyenne simple, poids nul = partenaire exclu, un seul partenaire, moyenne pondérée de deux partenaires, invariance d'échelle) et `S2sConfig::reliability_low/high/floor` (poids d'un partenaire = rampe lisse de sa dernière concordance connue, 1 avant toute information ; `reliability_weight` testée exactement). Option **désactivée par défaut**. Mutation : 6 mutants sur 9 détectés ; survivants : `>=` au lieu de `>` à l'activation (équivalent : des seuils égaux donnent des poids tous à 1), et deux vrais manques de test (indexation `scores[décalage du stack + coupe]` du partenaire ; score mis à jour à `max(avant, après)` plutôt que `avant`), non comblés car l'option est expérimentale et jugée nuisible.

Banc d'essai (calme 2, perturbation indépendante ±4 mm/°, flou σ = 1, 3 balayages, 2 niveaux ; part des coupes à ≤ 1,5 mm) :
- référence = notre propre sortie précédente, graine 7 : sans pondération 49 % ; seuils (0,5 ; 0,8) : 10 % ; seuils (0,6 ; 0,9) : 0 %.
- référence **indépendante (SVRTK)**, trois graines (7, 11, 13) : sans pondération 18 / 30 / 59 % (moyenne 36 %) ; seuils (0,3 ; 0,6) : 44 / 23 / 0 % (moyenne 22 %).

Pas de gain robuste : un gain sur une graine, une perte totale sur une autre. **Explication** : la concordance d'une coupe est une propriété de la PAIRE (coupe, partenaires). Quand la majorité des partenaires est fausse, une coupe correcte croisant des partenaires faux a une concordance basse, comme une coupe fausse : la concordance n'identifie pas les partenaires fiables sans consensus majoritaire, et pondérer renforce les partenaires qui ont par hasard une concordance élevée (auto-renforcement). Même un oracle parfait (section 5) ne rend la sélection correcte que dans 25 % des cas.

**Enseignements de méthode** :
- la part de coupes retrouvées à ±4 mm varie énormément avec la graine de la perturbation (24 à 63 % pour la même méthode) : **comparer des variantes exige plusieurs graines** (au moins trois) ; une seule perturbation ne permet de conclure que pour des écarts très grands ;
- une référence qui est la sortie d'une version de la méthode testée est **circulaire** (et changer le fichier de référence change les chiffres : 37 % → 49 % pour la même configuration) ; préférer une référence indépendante (SVRTK) et la figer ;
- les tableaux de portée de capture de la section « Un flou adapté aux données réelles » du livre sont des tirages uniques avec une référence de type notre sortie : les grands écarts (11 → 72 % à 3 mm) restent crédibles, les petits non.

Pistes restantes : (B) multi-départ au niveau du bloc ou du stack avec rotations (FLIRT) ; (C) régularisation temporelle ; (A') agrégation robuste des corrélations de paires (M-estimateur, moyenne tronquée : « la coupe s'accorde avec un sous-ensemble cohérent de partenaires », sans estimer de fiabilité).

## 7. Piste B (multi-départ au niveau du bloc, à la FLIRT) : pas de gain (2026-10-16)

Implémentation : `register_block_multistart` (grille de rotations {−d, 0, +d}³ autour du pivot du bloc ; à chaque rotation, translations seules optimisées par Adam sur une coupe du bloc sur trois ; raffinement à six paramètres des `keep` meilleurs ; gardé seulement si son coût dépasse celui de la recherche actuelle) ; `S2sConfig::block_multistart_deg/keep` (désactivé par défaut, non exposé en ligne de commande) ; tests (`block_cost_subset`, jamais pire que zéro, retrouve un décalage de 3 mm).

Banc d'essai « décalage de stack entier » (nouveau mode `MEDOXIDE_PERTURBATION=stack` : un mouvement rigide par stack, ± A mm et ± A °, autour de son barycentre cérébral ; calme 2, référence SVRTK, trois graines 7 / 11 / 13, flou σ = 1) :

| Amplitude A | sans alignement par blocs | alignement par blocs actuel (recherche ±12 mm) |
|---|---|---|
| ±4 | 85 / 96 / 11 % | 96 / 96 / 90 % |
| ±8 | 3 / 76 / 0 % | 94 / 94 / **0 %** |

**Ce que vaut « ±A »** : pour la graine 13 à ±8, les perturbations mesurées sont des rotations de 10-11° et des translations de **19-20 mm** pour deux stacks (13-16 mm pour les graines 7 et 11) : un décalage bien au-delà de la grille de translation de ±12 mm de la recherche actuelle.

Variantes testées sur la graine 13 :
- ±6 mm : recherche actuelle 86 % ; avec multi-départs (grille ±6°, 2 raffinés) 79 % ; recherche élargie à ±20 mm 86 % (identique à l'actuelle) ;
- ±8 mm : recherche actuelle 0 % ; multi-départs 0 % ; recherche élargie à ±20 mm et ±28 mm : 7 % chacune (identiques entre elles) ; coût : ces recherches élargies prennent plus d'une heure chacune (grille de translation en cube).

**Conclusion** : l'alignement par stack actuel suffit pour les décalages réalistes (jusqu'à ±8 sur deux graines sur trois, c'est-à-dire des translations de 13 à 16 mm) ; ni le multi-départ ni l'élargissement de la grille ne rattrapent le cas extrême (graine 13 à ±8), où le **coût lui-même** préfère une mauvaise solution (les stacks, décalés de ~20 mm et ~11°, se recouvrent à peine). Le multi-départ a même rendu un résultat un peu moins bon à ±6 (79 contre 86 %) : une pose de concordance supérieure mais plus éloignée de la référence. Le critère de départ (±8 mm : trois graines ≥ 90 %) n'est donc pas atteint, et la cause n'est pas la rotation.

Pistes restantes : (A') agrégation robuste des corrélations de paires ; (C) régularisation temporelle ; et pour les décalages très grands une initialisation non fondée sur l'intensité seule (géométrie : chevauchement des masques, barycentres) plutôt qu'une grille plus large.

## 8. Piste C (régularisation temporelle) : critères fixés avant l'évaluation (2026-10-16)

Brique : `IntersectionProblem::with_temporal_prior` + `S2sConfig::temporal_weight` (λ en 1/mm²) et `temporal_points` (64) : le coût d'une coupe devient `concordance − λ · moyenne |P' − Q|²`, `P'` = 64 pixels de masque de la coupe après correction, `Q` = leurs positions selon le mouvement relatif des voisines directes (coupe précédente et suivante du stack, si recalables). Désactivée par défaut.

Évaluation : banc de portée (calme 2, référence SVRTK, trois graines 7 / 11 / 13, flou σ = 1) ; variantes λ = 0 (référence), 0,02, 0,1.
- **Succès** : perturbation lisse ±6 mm : au moins +10 points en moyenne sur trois graines ; ±4 mm : pas de perte de plus de 5 points ; puis (meilleur λ) perturbation indépendante ±3 mm : pas de perte de plus de 10 points ;
- **réel** (six sujets depuis l'en-tête) : aucune baisse de concordance supérieure à 0,01 par sujet, moyenne en hausse d'au moins 0,01, accord des poses non dégradé ; temps ≤ +10 %.

## 9. Recherche hors IRM fœtale : formalisme, optimisation, évaluation (2026-10-16)

Sources consultées (web) : état de l'art du recalage coupe-volume (Ferrante & Paragios, *Med. Image Anal.* 2017, arXiv 1702.01636, texte intégral lu) ; correction du mouvement intra-volume en IRM fonctionnelle (SLOMOCO, Beall & Lowe ; SLIMM, Sui et al., *NeuroImage* 2020 ; recalage coupe-volume par attention, arXiv 2404.04546) ; IRM de diffusion (`eddy` de FSL, Andersson et al., *NeuroImage* 2017, « within volume movement » ; SHARD diffusion, arXiv 1905.02996) ; suivi de mouvement par filtre de Kalman couplé au recalage coupe-volume (Marami et al. ; Afacan, Warfield : diffusion et IRM fœtale) ; échographie à main levée sans capteur (Prevost et al. 2018 ; modèles séquentiels, arXiv 2211.04867) ; reconstruction 3D d'histologie (Pichat et al. 2018, *A survey of methods for 3D histology reconstruction* ; reconstruction globale guidée par la régularité, *NeuroImage* 2011 ; recalage conjoint avec synthèse, Iglesias et al. 2018) ; recalage 3D–2D en radiologie interventionnelle (CMA-ES et multi-départs : PMC4915373 ; base d'optimisation CMA-ES, arXiv 2402.05642) ; robotique et vision (non-convexité graduelle, Yang et al. 2020, arXiv 1909.08605 ; moyennage de rotations et synchronisation, arXiv 2102.05454).

### Formalisme
- **F1. Trajectoire de mouvement à base temporelle réduite** (eddy, `--mporder`) : le mouvement intra-volume n'est pas libre à 6 paramètres par coupe mais exprimé dans une base de cosinus discrets (DCT) sur l'ordre d'acquisition, d'ordre de N/4 à N/2 (N = nombre d'excitations), avec régularisation (`--s2v_lambda` = 1 à 10, 5 itérations par défaut) ; l'ordre d'acquisition des coupes est une donnée d'entrée. Nos blocs hiérarchiques sont une approximation constante par morceaux de la même idée ; la base DCT (ou B-splines) est la version continue, et elle fournit une hiérarchie naturelle en fréquence. Limite : un mouvement par à-coups (sauts) n'est pas bien représenté par une base lisse.
- **F2. Espace d'états / filtre de Kalman le long du stack** (Marami et al., Afacan, Warfield ; fœtal : chaîne de Markov cachée) : « placer des contraintes physiques sur le mouvement de coupe à coupe plutôt que de le supposer arbitraire », avec un filtre robuste aux valeurs aberrantes. Notre pénalité temporelle est le maximum a posteriori d'une marche aléatoire de poids λ fixe ; un lisseur de Kalman (Rauch–Tung–Striebel) la généralise : **le poids de chaque coupe vient de l'incertitude de sa mesure** (courbure de la concordance), donc une coupe bien contrainte par ses partenaires suit ses données, une coupe ambiguë suit ses voisines. C'est une réponse à l'échec de la pondération par la concordance (la fiabilité vient de l'information, pas du niveau du score).
- **F3. Graphe de poses et moyennage de rotations** (vision, robotique) : contraintes relatives par paires, solveurs globaux, cohérence de cycles, noyaux robustes. Nos contraintes (accord de profils d'intensité) ne sont pas des poses relatives mesurées : transposition indirecte, priorité faible, mais la notion de **cohérence de cycles** pourrait détecter les coupes incohérentes.
- **F4. Modèle d'intensité conjoint** (histologie : recalage et synthèse d'intensité estimés ensemble, modèle probabiliste ; fMRI/diffusion : modèle de prédiction par processus gaussien) : cohérent avec ce que nous avons fait pour le biais ; piste : estimer conjointement gain par coupe et pose.
- **F5. Cohérence intra-stack** (histologie : reconstruction globale par régression des intensités de coupes voisines, qui évite l'effet « banane » et la dérive en z des recalages par paires successives) : régularisation de forme et d'intensité entre coupes voisines ; pertinent si les coupes voisines d'un stack se recouvrent (épaisseur > pas) ; à vérifier.

### Optimisation
- **O1. CMA-ES avec multi-départs partitionnés et redémarrages** (3D–2D, PMC4915373 : espace de recherche partitionné en sous-espaces par kd-tree, une recherche CMA-ES par sous-espace, redémarrage final ; jusqu'à 250 départs, succès 99,99 %, 22 000 évaluations par seconde sur GPU ; base CMA-ES en 2024). L'état de l'art coupe-volume cite CMA-ES comme optimiseur global usuel. Notre diagnostic a montré que notre difficulté est la **sélection** (le coût d'une coupe contre des partenaires faux est trompeur), pas la génération de candidats : CMA-ES seul ne la résout pas. Garder en réserve ; retenir la **méthode d'évaluation** (O1-bis ci-dessous).
- **O2. Non-convexité graduelle (GNC) avec dualité de Black–Rangarajan** (Yang et al. 2020) : estimation robuste (Geman–McClure ou moindres carrés tronqués) par alternance « moindres carrés pondérés » / « mise à jour des poids », avec un paramètre qui passe de la convexité à la non-convexité ; **tolère 70 à 80 % de valeurs aberrantes sans initialisation** et surpasse RANSAC. C'est la formalisation propre de notre piste A' : le poids est celui de chaque **paire** (résidu de la paire), pas celui d'une coupe (ce qui échouait parce que la concordance est une propriété de la paire). Priorité haute.
- **O3. Gauss–Newton / Levenberg–Marquardt à gradient analytique** (l'état de l'art note que Levenberg–Marquardt est plus robuste que Gauss–Newton et que Nelder–Mead/Powell sont des compromis robustesse/temps) : nous utilisons Adam à pas décroissant ; un pas de second ordre approché (Gauss–Newton sur les résidus de profils) pourrait converger en peu d'itérations et fournir la **matrice d'information** utile à F2.
- **O4. Détection et remplacement des coupes aberrantes** (eddy : prédiction de chaque coupe par un processus gaussien, détection à ± 4 écarts types, remplacement) : analogue de notre seuil d'intensité ; piste : détecter les coupes en perte de signal par leur résidu et non seulement leur intensité.
- **O5. Multirésolution à pas liés à la résolution** (FLIRT, section 1) : déjà adopté en esprit (flou σ = 1 puis 0,5).

### Évaluation
- **E1. Protocole standard confirmé** (état de l'art, section 7) : vérité terrain synthétique (coupes extraites avec des transformations connues, paramètres perturbés comme initialisation) et « étalon de bronze » (poses d'une méthode plus fiable) ; indicateurs : distance entre paramètres, erreur sur points cibles (TRE), différence d'intensité avant/après. C'est ce que fait notre banc d'essai (référence de consensus). Le même article déplore **l'absence de banc d'essai public** pour le recalage coupe-volume (aucun jeu conçu pour cela).
- **E1-bis. Taux de succès en fonction de l'amplitude de la perturbation, avec beaucoup d'essais** (3D–2D : 100 à 200 essais par configuration, 7 200 à 14 400 recalages ; critère de succès < 5 mm ; portée de capture définie par ce taux). Notre banc n'a que 3 graines par configuration, avec une variabilité de ± 15 points : il faut **plus d'essais** (graines, et variantes d'amplitude) et des courbes de succès plutôt que des points.
- **E2. Données semi-synthétiques à contenu réel** (fMRI : comparaison du mouvement inter- et intra-volume avec des données de mouvement simulé « étalon or » ; histologie/IRM : fantômes mixtes) : appliquer un mouvement connu à un volume RÉEL pour simuler les coupes. Notre simulateur d'atlas est trop lisse (la portée réelle est deux fois plus petite) : **simuler à partir d'un volume réel reconstruit** (par exemple la reconstruction d'un sujet calme à 0,8 mm), avec notre modèle d'acquisition (PSF, épaisseur, bruit), des mouvements connus (indépendants, lisses, à sauts), le biais d'intensité et la perte de signal : vérité exacte et contenu réaliste. Priorité haute.
- **E3. Qualité anatomique indirecte** (pas de repères en IRM fœtale) : concordance et accord entre méthodes (déjà), prédiction d'une coupe ou d'un stack mis de côté (déjà, invalide en présence de correction de biais), et recouvrement de segmentations (volume cérébral, ventricules) entre reconstructions : à ajouter.

## 10. Plan d'implémentation mis à jour (2026-10-16)

État : A (pondération des partenaires) rejetée ; B (multi-départ de bloc) sans gain ; **C (régularisation temporelle) très prometteuse** (banc lisse ±6 mm : 23 % → 90 % ; indépendant ±3 mm : 66 % → 88 % ; accord avec les méthodes indépendantes meilleur sur tous les sujets réels, la concordance baissant un peu comme attendu pour un a priori de lissage).

Étapes, dans l'ordre (une à la fois, pause et critères fixés avant chacune) :
1. **Finir C** : choisir λ (0,005 ou 0,02), décider du défaut avec l'utilisateur, tests de mutation de la brique, documentation.
2. **E2 : banc semi-synthétique à contenu réel** (volume réel reconstruit → coupes simulées avec mouvement connu : indépendant, lisse, à sauts ; biais ; perte de signal) et **E1-bis : courbes de taux de succès** avec davantage d'essais. C'est l'outil qui permet de trancher les suivantes avec une vérité exacte et du contenu réaliste.
3. **F2 : lisseur de Kalman le long du stack** : poids temporel par coupe issu de l'information (courbure de la concordance) à la place d'un λ fixe ; critère : meilleur que le meilleur λ fixe sur le banc E2 (indépendant, lisse, sauts) et sur l'accord réel.
4. **O2 : agrégation robuste par non-convexité graduelle** des corrélations de paires (poids par paire) ; critère : gain sur le banc E2 au-delà de C, sans perte sur les sujets calmes.
5. **F1 : base temporelle réduite (DCT ou B-splines)** comme alternative aux blocs et à λ ; critère : égal ou meilleur que C avec moins de paramètres, y compris avec des sauts.
6. En réserve : O1 (CMA-ES en population, si la sélection des candidats devient fiable grâce à 3 et 4), O3 (pas de second ordre), O4 (détection des coupes aberrantes par résidu), F5 (cohérence intra-stack), F3 (cohérence de cycles), E3 (recouvrement de segmentations).
