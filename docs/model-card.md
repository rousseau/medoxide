---
license: cc-by-4.0
tags:
  - medical
  - mri
  - fetal-mri
  - brain-extraction
  - segmentation
  - burn
  - rust
---

# Fetal-BET pour Burn (medoxide)

Poids de **Fetal-BET** (Attention U-Net 2D pour l'extraction du cerveau en IRM
fœtale), convertis au format `burnpack` de [Burn](https://burn.dev) pour la
commande `medx fetalbet` de [medoxide](https://github.com/rousseau/medoxide).

## Modèle d'origine

Ce modèle est **Fetal-BET**, conçu et entraîné par Razieh Faghihpirayesh, Davood
Karimi, Deniz Erdoğmuş et Ali Gholipour. Ce dépôt n'en est qu'une **conversion de
format** : rien n'a été réentraîné.

- **Article** : Faghihpirayesh R., Karimi D., Erdoğmuş D., Gholipour A.,
  *Fetal-BET: Brain Extraction Tool for Fetal MRI*, IEEE Open Journal of
  Engineering in Medicine and Biology, 2024.
- **Dépôt d'origine** : <https://github.com/IntelligentImaging/fetal-brain-extraction>
- **Licence d'origine** : [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Ces poids sont redistribués sous la **même licence**.

La conversion n'a pas été réalisée par les auteurs d'origine et n'engage pas leur
responsabilité.

## Fichier

| Fichier | Taille | SHA-256 |
|---|---|---|
| `attunet.bpk` | 126 942 984 octets | `a70bcbe8da5f791b751c293a851505eb1aa9aa44c50def0afc41fd34bd60c3c2` |

## Utilisation

```bash
hf download rousseau/medoxide-fetalbet attunet.bpk --local-dir models
# ou, sans outil particulier :
curl -L -o models/attunet.bpk https://huggingface.co/rousseau/medoxide-fetalbet/resolve/main/attunet.bpk

./target/release/medx fetalbet --input mon_image.nii.gz --output mon_masque.nii.gz
```

Documentation de medoxide (installation, options) : <https://rousseau.github.io/medoxide/>.

## Provenance et modifications

- **Source** : `AttUNet.pth` (SHA-256 `dd64c4d2852a1a68d8c0bd420a6b5275993fc41c0766fc561fbbacd50a192772`),
  extrait de l'image Docker `faghihpirayesh/fetal-bet` publiée par les auteurs
  (`/app/src/saved_models/AttUNet.pth`).
- **Modifications** : export ONNX (opset 17), puis conversion par `onnx2burn`
  (burn-onnx 0.22.0-pre.4) en poids `burnpack`. Les calculs sont équivalents à
  l'original : écart relatif des logits de 8e-6 entre PyTorch et ONNX, et de 3e-6
  entre Burn (wgpu) et onnxruntime sur une tuile de test.
- **Validation** : sur 8 volumes d'exemple du dépôt Fetal-BET, le masque produit
  par `medx fetalbet` a un Dice d'au moins 0,9998 avec celui de Fetal-BET.

## Licence et citation

Comme Fetal-BET, ces poids sont sous licence
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) : vous pouvez les
réutiliser en citant les auteurs d'origine, en indiquant la licence et les
modifications ci-dessus. Merci de citer l'article de Fetal-BET :

```bibtex
@article{faghihpirayesh2024fetal,
  title={Fetal-bet: Brain extraction tool for fetal mri},
  author={Faghihpirayesh, Razieh and Karimi, Davood and Erdo{\u{g}}mu{\c{s}}, Deniz and Gholipour, Ali},
  journal={IEEE Open Journal of Engineering in Medicine and Biology},
  year={2024},
  publisher={IEEE}
}
```

## Limites

Ces poids sont destinés à la **recherche**, pas à un usage médical ou diagnostique,
et sont fournis sans garantie, comme le précisent les auteurs de Fetal-BET. La
validation porte sur 8 volumes d'un seul jeu de données ; elle compare à Fetal-BET
et non à une segmentation manuelle.
