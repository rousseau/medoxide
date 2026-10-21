//! Réseau **43 étiquettes** de Multi-BOUNTI (fond et 43 étiquettes de tissus), généré par `onnx2burn` (burn-onnx 0.22.0-pre.4) à partir de `data/multibounti/onnx/multi_64.onnx`.
//!
//! # Attribution
//!
//! Ce modèle fait partie de la chaîne officielle de Multi-BOUNTI (segmentation du cerveau périnatal en 43 étiquettes) :
//!
//! > Uus A. U. et al., résumé « Multi-BOUNTI », ISMRM 2026, Le Cap (9 au 14 mai 2026) : <https://echo.ismrm.org/abstracts/view/853f2f75-843f-44d0-b5dc-4bcba456176d>.
//! > Code : <https://github.com/SVRTK/auto-proc-svrtk> ; image Docker `fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd`. Voir aussi BOUNTI (Uus et al., *eLife*, 2023, doi:10.7554/eLife.88818).
//!
//! Le code de Multi-BOUNTI et ses poids sont distribués sous licence **GNU GPL v3** ; les poids ne sont **pas** dans ce dépôt (licence MIT) : ils se régénèrent avec
//! `scripts/export_multibounti_onnx.py multi` puis `onnx2burn` (voir `docs/MULTIBOUNTI.md`). Poids : `patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026`. L'architecture est celle de la classe `AttentionUnet` de MONAI (Apache-2.0)
//! avec les réglages du script d'inférence officiel : 1 canal d'entrée, **44 sorties**, canaux 32 à 512 : c'est exactement celle de [`crate::bet`], seul le nombre de sorties de la dernière convolution change.
//!
//! **Modifications** : les poids PyTorch d'origine ont été exportés en ONNX (opset 17) puis convertis en code Rust pour Burn et en poids au format `burnpack`.
//! Ce fichier est le résultat de cette conversion, avec un seul changement à la main : l'import de `InstanceNorm`, remplacé par `crate::norme::InstanceNorm` (réductions en cascade, précis à grande taille).
//!
//! Les auteurs destinent l'outil à la recherche, pas à un usage médical ou diagnostique ; il est fourni sans garantie.
#![allow(clippy::all)]
// `from_bytes` et `Default` (chemin de poids du générateur) sont générés mais pas utilisés ici.
#![allow(dead_code)]

// Generated from ONNX "data/multibounti/onnx/multi_64.onnx" by burn-onnx
extern crate alloc;
use crate::norme::InstanceNorm;
use crate::norme::InstanceNormConfig;
use burn::nn::conv::Conv3d;
use burn::nn::conv::Conv3dConfig;
use burn::nn::conv::ConvTranspose3d;
use burn::nn::conv::ConvTranspose3dConfig;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::PRelu;
use burn::nn::PReluConfig;
use burn::nn::PaddingConfig3d;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Model {
    conv3d1: Conv3d,
    conv3d2: Conv3d,
    conv3d3: Conv3d,
    conv3d4: Conv3d,
    conv3d5: Conv3d,
    conv3d6: Conv3d,
    conv3d7: Conv3d,
    conv3d8: Conv3d,
    conv3d9: Conv3d,
    conv3d10: Conv3d,
    convtranspose3d1: ConvTranspose3d,
    batchnormalization1: BatchNorm,
    conv3d11: Conv3d,
    conv3d12: Conv3d,
    conv3d13: Conv3d,
    conv3d14: Conv3d,
    instancenormalization1: InstanceNorm,
    prelu1: PRelu,
    convtranspose3d2: ConvTranspose3d,
    batchnormalization2: BatchNorm,
    conv3d15: Conv3d,
    conv3d16: Conv3d,
    conv3d17: Conv3d,
    conv3d18: Conv3d,
    instancenormalization2: InstanceNorm,
    prelu2: PRelu,
    convtranspose3d3: ConvTranspose3d,
    batchnormalization3: BatchNorm,
    conv3d19: Conv3d,
    conv3d20: Conv3d,
    conv3d21: Conv3d,
    conv3d22: Conv3d,
    instancenormalization3: InstanceNorm,
    prelu3: PRelu,
    convtranspose3d4: ConvTranspose3d,
    batchnormalization4: BatchNorm,
    conv3d23: Conv3d,
    conv3d24: Conv3d,
    conv3d25: Conv3d,
    conv3d26: Conv3d,
    instancenormalization4: InstanceNorm,
    prelu4: PRelu,
    conv3d27: Conv3d,
    #[module(skip)]
    device: Device,
}

extern crate std;

impl Default for Model {
    fn default() -> Self {
        Self::from_file("data/bounti/burn/bet/bet_64.bpk", &Default::default())
    }
}

impl Model {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(&file);
        model
            .load_from(&mut store)
            .unwrap_or_else(|e| panic!("Failed to load burnpack file {}: {e}", file.as_ref().display()));
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
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d2 = Conv3dConfig::new([32, 32], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d3 = Conv3dConfig::new([32, 64], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d4 = Conv3dConfig::new([64, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d5 = Conv3dConfig::new([64, 128], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d6 = Conv3dConfig::new([128, 128], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d7 = Conv3dConfig::new([128, 256], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d8 = Conv3dConfig::new([256, 256], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d9 = Conv3dConfig::new([256, 512], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d10 = Conv3dConfig::new([512, 512], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose3d1 = ConvTranspose3dConfig::new([512, 256], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv3d11 = Conv3dConfig::new([256, 128], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d12 = Conv3dConfig::new([256, 128], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d13 = Conv3dConfig::new([128, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d14 = Conv3dConfig::new([512, 256], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization1 = InstanceNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu1 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d2 = ConvTranspose3dConfig::new([256, 128], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv3d15 = Conv3dConfig::new([128, 64], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d16 = Conv3dConfig::new([128, 64], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d17 = Conv3dConfig::new([64, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d18 = Conv3dConfig::new([256, 128], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization2 = InstanceNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu2 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d3 = ConvTranspose3dConfig::new([128, 64], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv3d19 = Conv3dConfig::new([64, 32], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d20 = Conv3dConfig::new([64, 32], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d21 = Conv3dConfig::new([32, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d22 = Conv3dConfig::new([128, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization3 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu3 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose3d4 = ConvTranspose3dConfig::new([64, 32], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding([1, 1, 1])
            .with_padding_out([1, 1, 1])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv3d23 = Conv3dConfig::new([32, 16], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d24 = Conv3dConfig::new([32, 16], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d25 = Conv3dConfig::new([16, 1], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d26 = Conv3dConfig::new([64, 32], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization4 = InstanceNormConfig::new(32)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu4 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv3d27 = Conv3dConfig::new([32, 44], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv3d1,
            conv3d2,
            conv3d3,
            conv3d4,
            conv3d5,
            conv3d6,
            conv3d7,
            conv3d8,
            conv3d9,
            conv3d10,
            convtranspose3d1,
            batchnormalization1,
            conv3d11,
            conv3d12,
            conv3d13,
            conv3d14,
            instancenormalization1,
            prelu1,
            convtranspose3d2,
            batchnormalization2,
            conv3d15,
            conv3d16,
            conv3d17,
            conv3d18,
            instancenormalization2,
            prelu2,
            convtranspose3d3,
            batchnormalization3,
            conv3d19,
            conv3d20,
            conv3d21,
            conv3d22,
            instancenormalization3,
            prelu3,
            convtranspose3d4,
            batchnormalization4,
            conv3d23,
            conv3d24,
            conv3d25,
            conv3d26,
            instancenormalization4,
            prelu4,
            conv3d27,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, image: Tensor<5>) -> Tensor<5> {
        let conv3d1_out1 = self.conv3d1.forward(image);
        let relu1_out1 = burn::tensor::activation::relu(conv3d1_out1);
        let conv3d2_out1 = self.conv3d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv3d2_out1);
        let conv3d3_out1 = self.conv3d3.forward(relu2_out1.clone());
        let relu3_out1 = burn::tensor::activation::relu(conv3d3_out1);
        let conv3d4_out1 = self.conv3d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv3d4_out1);
        let conv3d5_out1 = self.conv3d5.forward(relu4_out1.clone());
        let relu5_out1 = burn::tensor::activation::relu(conv3d5_out1);
        let conv3d6_out1 = self.conv3d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv3d6_out1);
        let conv3d7_out1 = self.conv3d7.forward(relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv3d7_out1);
        let conv3d8_out1 = self.conv3d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv3d8_out1);
        let conv3d9_out1 = self.conv3d9.forward(relu8_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv3d9_out1);
        let conv3d10_out1 = self.conv3d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv3d10_out1);
        let convtranspose3d1_out1 = self.convtranspose3d1.forward(relu10_out1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(convtranspose3d1_out1);
        let relu11_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let conv3d11_out1 = self.conv3d11.forward(relu11_out1.clone());
        let conv3d12_out1 = self.conv3d12.forward(relu8_out1.clone());
        let add1_out1 = conv3d11_out1.add(conv3d12_out1);
        let relu12_out1 = burn::tensor::activation::relu(add1_out1);
        let conv3d13_out1 = self.conv3d13.forward(relu12_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv3d13_out1);
        let mul1_out1 = relu8_out1.mul(sigmoid1_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([mul1_out1, relu11_out1].into(), 1);
        let conv3d14_out1 = self.conv3d14.forward(concat1_out1);
        let instancenormalization1_out1 = self.instancenormalization1.forward(conv3d14_out1);
        let prelu1_out1 = self.prelu1.forward(instancenormalization1_out1);
        let convtranspose3d2_out1 = self.convtranspose3d2.forward(prelu1_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(convtranspose3d2_out1);
        let relu13_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv3d15_out1 = self.conv3d15.forward(relu13_out1.clone());
        let conv3d16_out1 = self.conv3d16.forward(relu6_out1.clone());
        let add2_out1 = conv3d15_out1.add(conv3d16_out1);
        let relu14_out1 = burn::tensor::activation::relu(add2_out1);
        let conv3d17_out1 = self.conv3d17.forward(relu14_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv3d17_out1);
        let mul2_out1 = relu6_out1.mul(sigmoid2_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([mul2_out1, relu13_out1].into(), 1);
        let conv3d18_out1 = self.conv3d18.forward(concat2_out1);
        let instancenormalization2_out1 = self.instancenormalization2.forward(conv3d18_out1);
        let prelu2_out1 = self.prelu2.forward(instancenormalization2_out1);
        let convtranspose3d3_out1 = self.convtranspose3d3.forward(prelu2_out1);
        let batchnormalization3_out1 = self.batchnormalization3.forward(convtranspose3d3_out1);
        let relu15_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv3d19_out1 = self.conv3d19.forward(relu15_out1.clone());
        let conv3d20_out1 = self.conv3d20.forward(relu4_out1.clone());
        let add3_out1 = conv3d19_out1.add(conv3d20_out1);
        let relu16_out1 = burn::tensor::activation::relu(add3_out1);
        let conv3d21_out1 = self.conv3d21.forward(relu16_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv3d21_out1);
        let mul3_out1 = relu4_out1.mul(sigmoid3_out1);
        let concat3_out1 = burn::tensor::Tensor::cat([mul3_out1, relu15_out1].into(), 1);
        let conv3d22_out1 = self.conv3d22.forward(concat3_out1);
        let instancenormalization3_out1 = self.instancenormalization3.forward(conv3d22_out1);
        let prelu3_out1 = self.prelu3.forward(instancenormalization3_out1);
        let convtranspose3d4_out1 = self.convtranspose3d4.forward(prelu3_out1);
        let batchnormalization4_out1 = self.batchnormalization4.forward(convtranspose3d4_out1);
        let relu17_out1 = burn::tensor::activation::relu(batchnormalization4_out1);
        let conv3d23_out1 = self.conv3d23.forward(relu17_out1.clone());
        let conv3d24_out1 = self.conv3d24.forward(relu2_out1.clone());
        let add4_out1 = conv3d23_out1.add(conv3d24_out1);
        let relu18_out1 = burn::tensor::activation::relu(add4_out1);
        let conv3d25_out1 = self.conv3d25.forward(relu18_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv3d25_out1);
        let mul4_out1 = relu2_out1.mul(sigmoid4_out1);
        let concat4_out1 = burn::tensor::Tensor::cat([mul4_out1, relu17_out1].into(), 1);
        let conv3d26_out1 = self.conv3d26.forward(concat4_out1);
        let instancenormalization4_out1 = self.instancenormalization4.forward(conv3d26_out1);
        let prelu4_out1 = self.prelu4.forward(instancenormalization4_out1);
        let conv3d27_out1 = self.conv3d27.forward(prelu4_out1);
        conv3d27_out1
    }
}
