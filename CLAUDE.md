# Instructions pour Claude Code sur ce dépôt

Contexte : la personne qui développe ce projet apprend Rust et Burn **en
même temps** qu'elle construit l'outil. L'objectif n'est pas d'obtenir du
code qui marche le plus vite possible, mais de comprendre chaque brique
posée. Priorité : la pédagogie, avant la vitesse.

## Mode de travail attendu

- **Expliquer avant d'écrire.** Avant d'introduire un concept Rust ou Burn
  nouveau (ownership, traits, lifetimes, tenseurs, autodiff, backends...),
  dire en 2-3 phrases ce que c'est et pourquoi on en a besoin ici,
  *avant* de l'utiliser dans le code — pas après, en commentaire.
- **Petits pas.** Une seule chose nouvelle à la fois. Un changement qui
  mélange trois concepts nouveaux doit être scindé en plusieurs étapes,
  même si ça prend plus de messages.
- **S'arrêter pour relecture.** Après chaque étape significative (un
  nouveau fichier, une nouvelle dépendance, une nouvelle fonction
  publique), marquer une pause et attendre une confirmation avant de
  continuer, plutôt que d'enchaîner plusieurs étapes d'un coup.
- **Ne pas ajouter de dépendance ou d'abstraction par anticipation.**
  Si une crate ou une structure n'est pas encore utilisée par du code
  réel, ne pas l'ajouter "parce qu'on en aura besoin plus tard". Le
  workspace actuel suit ce principe (pas de `medoxide-core` tant qu'un
  deuxième module n'en justifie pas un).
- **Documenter en rustdoc au fil de l'eau**, pas après coup : toute
  fonction publique reçoit son commentaire `///` au moment où elle est
  écrite.
- **Mettre à jour `docs/LEARNING.md`** après chaque étape qui introduit un
  concept Rust/Burn nouveau pour la première fois dans le projet : une
  entrée courte (quoi, pourquoi, où dans le code).

## Validation

Chaque module se construit contre une référence Python existante, avec un
critère chiffré défini *avant* de coder (voir tableau "Roadmap" du
README). On ne passe à l'étape suivante que quand le critère de l'étape
en cours est atteint et vérifié — pas sur une impression que "ça a l'air
de marcher".

## Ce qu'il ne faut pas faire

- Ne pas générer de gros morceaux de code non expliqués "pour gagner du
  temps".
- Ne pas introduire d'outillage lourd (générateur de doc, CI complexe,
  frameworks de test additionnels) sans que le besoin soit concret.
- Ne pas avancer sur le module suivant si le critère de validation du
  module en cours n'est pas encore vérifié.
