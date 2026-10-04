//! Attention U-Net 2D de Fetal-BET, généré par `onnx2burn` (burn-onnx 0.22.0-pre.4)
//! à partir de `models/attunet.onnx`. Ne pas modifier à la main : régénérer avec
//! `onnx2burn models/attunet.onnx <dossier>` puis recopier `attunet.rs` ici.
//!
//! Les poids (`attunet.bpk`, 121 Mo) ne sont pas dans Git : voir Garage,
//! `medoxide-dev/models/`.
#![allow(clippy::all)]

extern crate alloc;
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::InstanceNorm;
use burn::nn::InstanceNormConfig;
use burn::nn::PRelu;
use burn::nn::PReluConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    conv2d10: Conv2d,
    convtranspose2d1: ConvTranspose2d,
    batchnormalization1: BatchNorm,
    conv2d11: Conv2d,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    instancenormalization1: InstanceNorm,
    prelu1: PRelu,
    convtranspose2d2: ConvTranspose2d,
    batchnormalization2: BatchNorm,
    conv2d15: Conv2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    instancenormalization2: InstanceNorm,
    prelu2: PRelu,
    convtranspose2d3: ConvTranspose2d,
    batchnormalization3: BatchNorm,
    conv2d19: Conv2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    instancenormalization3: InstanceNorm,
    prelu3: PRelu,
    convtranspose2d4: ConvTranspose2d,
    batchnormalization4: BatchNorm,
    conv2d23: Conv2d,
    conv2d24: Conv2d,
    conv2d25: Conv2d,
    conv2d26: Conv2d,
    instancenormalization4: InstanceNorm,
    prelu4: PRelu,
    conv2d27: Conv2d,
    #[module(skip)]
    device: Device,
}


extern crate std;

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
        let conv2d1 = Conv2dConfig::new([1, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([512, 1024], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([1024, 1024], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([1024, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d11 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([256, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([1024, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization1 = InstanceNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu1 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose2d2 = ConvTranspose2dConfig::new([512, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d15 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([128, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([512, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization2 = InstanceNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu2 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose2d3 = ConvTranspose2dConfig::new([256, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d19 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([128, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([64, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([256, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization3 = InstanceNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu3 = PReluConfig::new().with_num_parameters(1).init(device);
        let convtranspose2d4 = ConvTranspose2dConfig::new([128, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d23 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([32, 1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization4 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu4 = PReluConfig::new().with_num_parameters(1).init(device);
        let conv2d27 = Conv2dConfig::new([64, 2], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            convtranspose2d1,
            batchnormalization1,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            instancenormalization1,
            prelu1,
            convtranspose2d2,
            batchnormalization2,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            instancenormalization2,
            prelu2,
            convtranspose2d3,
            batchnormalization3,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            instancenormalization3,
            prelu3,
            convtranspose2d4,
            batchnormalization4,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            instancenormalization4,
            prelu4,
            conv2d27,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, image: Tensor<4>) -> Tensor<4> {
        let conv2d1_out1 = self.conv2d1.forward(image);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1.clone());
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1.clone());
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(relu10_out1);
        let batchnormalization1_out1 = self
            .batchnormalization1
            .forward(convtranspose2d1_out1);
        let relu11_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu11_out1.clone());
        let conv2d12_out1 = self.conv2d12.forward(relu8_out1.clone());
        let add1_out1 = conv2d11_out1.add(conv2d12_out1);
        let relu12_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu12_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d13_out1);
        let mul1_out1 = relu8_out1.mul(sigmoid1_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([mul1_out1, relu11_out1].into(), 1);
        let conv2d14_out1 = self.conv2d14.forward(concat1_out1);
        let instancenormalization1_out1 = self
            .instancenormalization1
            .forward(conv2d14_out1);
        let prelu1_out1 = self.prelu1.forward(instancenormalization1_out1);
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(prelu1_out1);
        let batchnormalization2_out1 = self
            .batchnormalization2
            .forward(convtranspose2d2_out1);
        let relu13_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu13_out1.clone());
        let conv2d16_out1 = self.conv2d16.forward(relu6_out1.clone());
        let add2_out1 = conv2d15_out1.add(conv2d16_out1);
        let relu14_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu14_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d17_out1);
        let mul2_out1 = relu6_out1.mul(sigmoid2_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([mul2_out1, relu13_out1].into(), 1);
        let conv2d18_out1 = self.conv2d18.forward(concat2_out1);
        let instancenormalization2_out1 = self
            .instancenormalization2
            .forward(conv2d18_out1);
        let prelu2_out1 = self.prelu2.forward(instancenormalization2_out1);
        let convtranspose2d3_out1 = self.convtranspose2d3.forward(prelu2_out1);
        let batchnormalization3_out1 = self
            .batchnormalization3
            .forward(convtranspose2d3_out1);
        let relu15_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu15_out1.clone());
        let conv2d20_out1 = self.conv2d20.forward(relu4_out1.clone());
        let add3_out1 = conv2d19_out1.add(conv2d20_out1);
        let relu16_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu16_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv2d21_out1);
        let mul3_out1 = relu4_out1.mul(sigmoid3_out1);
        let concat3_out1 = burn::tensor::Tensor::cat([mul3_out1, relu15_out1].into(), 1);
        let conv2d22_out1 = self.conv2d22.forward(concat3_out1);
        let instancenormalization3_out1 = self
            .instancenormalization3
            .forward(conv2d22_out1);
        let prelu3_out1 = self.prelu3.forward(instancenormalization3_out1);
        let convtranspose2d4_out1 = self.convtranspose2d4.forward(prelu3_out1);
        let batchnormalization4_out1 = self
            .batchnormalization4
            .forward(convtranspose2d4_out1);
        let relu17_out1 = burn::tensor::activation::relu(batchnormalization4_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu17_out1.clone());
        let conv2d24_out1 = self.conv2d24.forward(relu2_out1.clone());
        let add4_out1 = conv2d23_out1.add(conv2d24_out1);
        let relu18_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu18_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv2d25_out1);
        let mul4_out1 = relu2_out1.mul(sigmoid4_out1);
        let concat4_out1 = burn::tensor::Tensor::cat([mul4_out1, relu17_out1].into(), 1);
        let conv2d26_out1 = self.conv2d26.forward(concat4_out1);
        let instancenormalization4_out1 = self
            .instancenormalization4
            .forward(conv2d26_out1);
        let prelu4_out1 = self.prelu4.forward(instancenormalization4_out1);
        let conv2d27_out1 = self.conv2d27.forward(prelu4_out1);
        conv2d27_out1
    }
}
