//! U-Net 3D de **BOUNTI** (segmentation en 19 tissus du cerveau fœtal), généré par `onnx2burn` (burn-onnx 0.22.0-pre.4) à partir de `data/bounti/onnx/unet_64.onnx`.
//!
//! # Attribution
//!
//! Ce modèle est l'un des deux réseaux de BOUNTI :
//!
//! > Uus A. U. et al., « BOUNTI: Brain vOlumetry and aUtomated parcellatioN for 3D feTal MRI »,
//! > *eLife*, 2023, doi:10.7554/eLife.88818. Code : <https://github.com/SVRTK/auto-proc-svrtk>
//!
//! Le code de BOUNTI et ses poids sont distribués sous licence **GNU GPL v3** ; les poids ne sont **pas** dans ce dépôt (licence MIT) : ils se
//! régénèrent avec `scripts/export_bounti_onnx.py` puis `onnx2burn` (voir `docs/BOUNTI.md`). L'architecture est celle de la classe `UNet` de MONAI
//! (Apache-2.0) avec les réglages du script d'inférence officiel : 1 canal d'entrée, 20 sorties, canaux 32 à 512, 1 unité résiduelle, PReLU, normalisation d'instance.
//!
//! **Modifications** : les poids PyTorch d'origine ont été exportés en ONNX (opset 17) puis convertis en code Rust pour Burn et en poids au format `burnpack`.
//! Ce fichier est le résultat de cette conversion, avec un seul changement à la main : l'import de `InstanceNorm`, remplacé par `crate::norme::InstanceNorm` (réductions en cascade, précis à 256³ ; voir `norme.rs`). Les calculs sont équivalents à l'original (voir les tests).
//!
//! Les auteurs destinent l'outil à la recherche, pas à un usage médical ou diagnostique ; il est fourni sans garantie.
//!
//! # Régénérer ce fichier
//!
//! `onnx2burn data/bounti/onnx/unet_64.onnx <dossier>`, puis recopier `unet_64.rs` ici **en conservant cet en-tête** (il n'est pas généré) et en remplaçant les deux imports `burn::nn::InstanceNorm` et `burn::nn::InstanceNormConfig` par `crate::norme::InstanceNorm` et `crate::norme::InstanceNormConfig`.
#![allow(clippy::all)]
// `from_bytes` et `Default` (chemin de poids du générateur) sont générés mais pas utilisés ici.
#![allow(dead_code)]

// Generated from ONNX "data/bounti/onnx/unet_64.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use crate::norme::InstanceNorm;
use crate::norme::InstanceNormConfig;
use burn::nn::PRelu;
use burn::nn::PReluConfig;
use burn::nn::PaddingConfig3d;
use burn::nn::conv::Conv3d;
use burn::nn::conv::Conv3dConfig;
use burn::nn::conv::ConvTranspose3d;
use burn::nn::conv::ConvTranspose3dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model {
    conv3d1: Conv3d,
    conv3d2: Conv3d,
    instancenormalization1: InstanceNorm,
    prelu1: PRelu,
    conv3d3: Conv3d,
    conv3d4: Conv3d,
    instancenormalization2: InstanceNorm,
    prelu2: PRelu,
    conv3d5: Conv3d,
    conv3d6: Conv3d,
    instancenormalization3: InstanceNorm,
    prelu3: PRelu,
    conv3d7: Conv3d,
    conv3d8: Conv3d,
    instancenormalization4: InstanceNorm,
    prelu4: PRelu,
    conv3d9: Conv3d,
    conv3d10: Conv3d,
    instancenormalization5: InstanceNorm,
    prelu5: PRelu,
    convtranspose3d1: ConvTranspose3d,
    instancenormalization6: InstanceNorm,
    prelu6: PRelu,
    conv3d11: Conv3d,
    instancenormalization7: InstanceNorm,
    prelu7: PRelu,
    convtranspose3d2: ConvTranspose3d,
    instancenormalization8: InstanceNorm,
    prelu8: PRelu,
    conv3d12: Conv3d,
    instancenormalization9: InstanceNorm,
    prelu9: PRelu,
    convtranspose3d3: ConvTranspose3d,
    instancenormalization10: InstanceNorm,
    prelu10: PRelu,
    conv3d13: Conv3d,
    instancenormalization11: InstanceNorm,
    prelu11: PRelu,
    convtranspose3d4: ConvTranspose3d,
    instancenormalization12: InstanceNorm,
    prelu12: PRelu,
    conv3d14: Conv3d,
    #[module(skip)]
    device: Device,
}


extern crate std;

impl Default for Model {
    fn default() -> Self {
        Self::from_file("data/bounti/burn/unet/unet_64.bpk", &Default::default())
    }
}

impl Model {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(&file);
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| {
                panic!("Failed to load burnpack file {}: {e}", file.as_ref().display())
            });
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| panic!("Failed to load burnpack bytes: {e}"));
        model
    }
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv3d1 = Conv3dConfig::new([1, 32], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d2 = Conv3dConfig::new([1, 32], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization1 = InstanceNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu1 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d3 = Conv3dConfig::new([32, 64], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d4 = Conv3dConfig::new([32, 64], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization2 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu2 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d5 = Conv3dConfig::new([64, 128], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d6 = Conv3dConfig::new([64, 128], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization3 = InstanceNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu3 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d7 = Conv3dConfig::new([128, 256], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d8 = Conv3dConfig::new([128, 256], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization4 = InstanceNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu4 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d9 = Conv3dConfig::new([256, 512], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d10 = Conv3dConfig::new([256, 512], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization5 = InstanceNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu5 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d1 = ConvTranspose3dConfig::new([768, 128], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization6 = InstanceNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu6 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d11 = Conv3dConfig::new([128, 128], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization7 = InstanceNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu7 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d2 = ConvTranspose3dConfig::new([256, 64], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization8 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu8 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d12 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization9 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu9 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d3 = ConvTranspose3dConfig::new([128, 32], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization10 = InstanceNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu10 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d13 = Conv3dConfig::new([32, 32], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization11 = InstanceNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu11 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d4 = ConvTranspose3dConfig::new([64, 20], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization12 = InstanceNormConfig::new(20)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu12 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d14 = Conv3dConfig::new([20, 20], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv3d1,
            conv3d2,
            instancenormalization1,
            prelu1,
            conv3d3,
            conv3d4,
            instancenormalization2,
            prelu2,
            conv3d5,
            conv3d6,
            instancenormalization3,
            prelu3,
            conv3d7,
            conv3d8,
            instancenormalization4,
            prelu4,
            conv3d9,
            conv3d10,
            instancenormalization5,
            prelu5,
            convtranspose3d1,
            instancenormalization6,
            prelu6,
            conv3d11,
            instancenormalization7,
            prelu7,
            convtranspose3d2,
            instancenormalization8,
            prelu8,
            conv3d12,
            instancenormalization9,
            prelu9,
            convtranspose3d3,
            instancenormalization10,
            prelu10,
            conv3d13,
            instancenormalization11,
            prelu11,
            convtranspose3d4,
            instancenormalization12,
            prelu12,
            conv3d14,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, image: Tensor<5>) -> Tensor<5> {
        let conv3d1_out1 = self.conv3d1.forward(image.clone());
        let conv3d2_out1 = self.conv3d2.forward(image);
        let instancenormalization1_out1 = self
            .instancenormalization1
            .forward(conv3d2_out1);
        let prelu1_out1 = self.prelu1.forward(instancenormalization1_out1);
        let add1_out1 = prelu1_out1.add(conv3d1_out1);
        let conv3d3_out1 = self.conv3d3.forward(add1_out1.clone());
        let conv3d4_out1 = self.conv3d4.forward(add1_out1.clone());
        let instancenormalization2_out1 = self
            .instancenormalization2
            .forward(conv3d4_out1);
        let prelu2_out1 = self.prelu2.forward(instancenormalization2_out1);
        let add2_out1 = prelu2_out1.add(conv3d3_out1);
        let conv3d5_out1 = self.conv3d5.forward(add2_out1.clone());
        let conv3d6_out1 = self.conv3d6.forward(add2_out1.clone());
        let instancenormalization3_out1 = self
            .instancenormalization3
            .forward(conv3d6_out1);
        let prelu3_out1 = self.prelu3.forward(instancenormalization3_out1);
        let add3_out1 = prelu3_out1.add(conv3d5_out1);
        let conv3d7_out1 = self.conv3d7.forward(add3_out1.clone());
        let conv3d8_out1 = self.conv3d8.forward(add3_out1.clone());
        let instancenormalization4_out1 = self
            .instancenormalization4
            .forward(conv3d8_out1);
        let prelu4_out1 = self.prelu4.forward(instancenormalization4_out1);
        let add4_out1 = prelu4_out1.add(conv3d7_out1);
        let conv3d9_out1 = self.conv3d9.forward(add4_out1.clone());
        let conv3d10_out1 = self.conv3d10.forward(add4_out1.clone());
        let instancenormalization5_out1 = self
            .instancenormalization5
            .forward(conv3d10_out1);
        let prelu5_out1 = self.prelu5.forward(instancenormalization5_out1);
        let add5_out1 = prelu5_out1.add(conv3d9_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([add4_out1, add5_out1].into(), 1);
        let convtranspose3d1_out1 = self.convtranspose3d1.forward(concat1_out1);
        let instancenormalization6_out1 = self
            .instancenormalization6
            .forward(convtranspose3d1_out1);
        let prelu6_out1 = self.prelu6.forward(instancenormalization6_out1);
        let conv3d11_out1 = self.conv3d11.forward(prelu6_out1.clone());
        let instancenormalization7_out1 = self
            .instancenormalization7
            .forward(conv3d11_out1);
        let prelu7_out1 = self.prelu7.forward(instancenormalization7_out1);
        let add6_out1 = prelu7_out1.add(prelu6_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([add3_out1, add6_out1].into(), 1);
        let convtranspose3d2_out1 = self.convtranspose3d2.forward(concat2_out1);
        let instancenormalization8_out1 = self
            .instancenormalization8
            .forward(convtranspose3d2_out1);
        let prelu8_out1 = self.prelu8.forward(instancenormalization8_out1);
        let conv3d12_out1 = self.conv3d12.forward(prelu8_out1.clone());
        let instancenormalization9_out1 = self
            .instancenormalization9
            .forward(conv3d12_out1);
        let prelu9_out1 = self.prelu9.forward(instancenormalization9_out1);
        let add7_out1 = prelu9_out1.add(prelu8_out1);
        let concat3_out1 = burn::tensor::Tensor::cat([add2_out1, add7_out1].into(), 1);
        let convtranspose3d3_out1 = self.convtranspose3d3.forward(concat3_out1);
        let instancenormalization10_out1 = self
            .instancenormalization10
            .forward(convtranspose3d3_out1);
        let prelu10_out1 = self.prelu10.forward(instancenormalization10_out1);
        let conv3d13_out1 = self.conv3d13.forward(prelu10_out1.clone());
        let instancenormalization11_out1 = self
            .instancenormalization11
            .forward(conv3d13_out1);
        let prelu11_out1 = self.prelu11.forward(instancenormalization11_out1);
        let add8_out1 = prelu11_out1.add(prelu10_out1);
        let concat4_out1 = burn::tensor::Tensor::cat([add1_out1, add8_out1].into(), 1);
        let convtranspose3d4_out1 = self.convtranspose3d4.forward(concat4_out1);
        let instancenormalization12_out1 = self
            .instancenormalization12
            .forward(convtranspose3d4_out1);
        let prelu12_out1 = self.prelu12.forward(instancenormalization12_out1);
        let conv3d14_out1 = self.conv3d14.forward(prelu12_out1.clone());
        let add9_out1 = conv3d14_out1.add(prelu12_out1);
        add9_out1
    }
}
