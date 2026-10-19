"""Lance une commande et l'arrête si sa mémoire résidente dépasse une limite ; affiche le pic de mémoire à la fin.

Pourquoi : un calcul en double précision à grande taille a déjà fait planter une machine de 24 Go. La limite s'applique au processus lancé et à ses descendants directs (somme des tailles résidentes lues par `ps`).

Usage : python scripts/garde_memoire.py <limite_Go> <commande> [arguments…]
Code de sortie : celui de la commande, ou 99 si la limite a été dépassée (la commande est alors tuée).
"""

import os
import signal
import subprocess
import sys
import time


def memoire_ko(pid):
    """Taille résidente (Ko) du processus `pid` et de ses descendants (via `ps`)."""
    sortie = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    enfants, tailles = {}, {}
    for ligne in sortie.splitlines():
        champs = ligne.split()
        if len(champs) == 3:
            p, pp, rss = map(int, champs)
            enfants.setdefault(pp, []).append(p)
            tailles[p] = rss
    total, pile = 0, [pid]
    while pile:
        courant = pile.pop()
        total += tailles.get(courant, 0)
        pile.extend(enfants.get(courant, []))
    return total


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    limite_ko = float(sys.argv[1]) * 1024 * 1024
    processus = subprocess.Popen(sys.argv[2:], start_new_session=True)
    pic = 0
    while processus.poll() is None:
        taille = memoire_ko(processus.pid)
        pic = max(pic, taille)
        if taille > limite_ko:
            os.killpg(processus.pid, signal.SIGKILL)
            processus.wait()
            print(f"GARDE-FOU : {taille / 1048576:.1f} Go > {limite_ko / 1048576:.1f} Go, commande arrêtée", flush=True)
            return 99
        time.sleep(0.5)
    print(f"pic de mémoire résidente : {pic / 1048576:.1f} Go", flush=True)
    return processus.returncode


if __name__ == "__main__":
    raise SystemExit(main())
