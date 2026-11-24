"""Plans de P1 (docs/ULF.md) pour les scripts de comparaison : les identifiants des sujets restent hors dépôt.

`MEDOXIDE_ULF_SUJETS` : identifiants des sujets de P1 séparés par des virgules (noms des dossiers `<sujet>/<plan>/` des
données) ; `--plans` vaut alors `<sujet>_<plan>` pour chaque sujet et les plans xy, yz, zx. Sans la variable, `--plans`
est obligatoire.
"""

import os

PLANS_PAR_SUJET = ("xy", "yz", "zx")


def option_plans(parseur):
    sujets = os.environ.get("MEDOXIDE_ULF_SUJETS")
    defaut = ",".join(f"{s}_{q}" for s in sujets.split(",") for q in PLANS_PAR_SUJET) if sujets else None
    parseur.add_argument("--plans", default=defaut, required=defaut is None,
                         help="<sujet>_<plan> séparés par des virgules (défaut : depuis MEDOXIDE_ULF_SUJETS)")
