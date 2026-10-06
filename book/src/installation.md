# Installation

medoxide se compile depuis les sources. Aucun binaire précompilé n'est fourni
pour l'instant.

## Prérequis

- **Rust**, installé via [rustup](https://rustup.rs). Burn 0.22.0-pre.4 demande
  Rust 1.95 ou plus récent ; la configuration du dépôt suit la version stable.
- **Un GPU compatible `wgpu`** : Metal sur Mac, Vulkan ou DX12 ailleurs. Seul un
  Mac avec GPU Apple a été testé.
- **Les poids du modèle** (`attunet.bpk`, 121 Mo), décrits plus bas.

## Compiler

```bash
git clone https://github.com/rousseau/medoxide.git
cd medoxide
cargo build --release
```

La première compilation télécharge et compile Burn et ses dépendances : compter
quelques minutes (environ 3 minutes sur notre machine). Le binaire est
`target/release/medx`. Pour vérifier qu'il fonctionne :

```bash
./target/release/medx --help
```

## Poids du modèle

Les poids ne sont pas dans le dépôt Git (121 Mo). `medx fetalbet` les cherche,
dans cet ordre :

1. l'option `--model chemin/attunet.bpk` ;
2. la variable d'environnement `MEDOXIDE_MODEL` ;
3. le fichier `models/attunet.bpk`, **relatif au dossier depuis lequel vous
   lancez la commande**.

Ils sont téléchargeables sur Hugging Face :
[rousseau/medoxide-fetalbet](https://huggingface.co/rousseau/medoxide-fetalbet).

```bash
mkdir -p models
curl -L -o models/attunet.bpk https://huggingface.co/rousseau/medoxide-fetalbet/resolve/main/attunet.bpk
```

Le SHA-256 attendu est `a70bcbe8da5f791b751c293a851505eb1aa9aa44c50def0afc41fd34bd60c3c2`
(`shasum -a 256 models/attunet.bpk`).

Ces poids sont ceux de Fetal-BET, convertis au format `burnpack` de Burn, sous la
même licence CC BY 4.0 : voir [Algorithmes et références](algorithmes.md) pour la
référence à citer. Vous pouvez aussi les régénérer vous-même :

### Les régénérer (avancé)

Chaîne exécutée une fois pendant le développement ; elle demande Docker, Python
(`torch`, `monai`, `onnx`, `onnxruntime`, `nibabel`) et Rust.

```bash
# 1. Extraire les poids PyTorch de l'image Docker de Fetal-BET (~16 Go de disque)
docker pull --platform linux/amd64 faghihpirayesh/fetal-bet
docker create --platform linux/amd64 --name fbe faghihpirayesh/fetal-bet
mkdir -p models && docker cp fbe:/app/src/saved_models/AttUNet.pth models/ && docker rm fbe

# 2. Exporter en ONNX et vérifier l'export (écart relatif < 1e-5, masques identiques)
python scripts/export_onnx.py

# 3. Convertir en poids Burn
cargo install burn-onnx --version 0.22.0-pre.4 --bin onnx2burn
onnx2burn models/attunet.onnx /dossier/de/sortie
cp /dossier/de/sortie/attunet.bpk models/
```

Le code du modèle (`crates/medoxide-fetalbet/src/model.rs`) vient de la même
commande `onnx2burn` : les poids doivent venir du même fichier ONNX.
