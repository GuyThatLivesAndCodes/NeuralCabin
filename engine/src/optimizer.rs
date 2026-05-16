//! Optimizer configuration and a thin step interface.
//!
//! The actual update math runs inside Burn (so it gets GPU acceleration); this
//! module just holds the user-chosen hyperparameters and constructs the Burn
//! optimizer when training starts. State buffers (momentum / variance) live
//! inside the Burn `OptimizerAdaptor` for the duration of a training run and
//! are not persisted.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OptimizerKind {
    Sgd { lr: f32, momentum: f32 },
    Adam { lr: f32, beta1: f32, beta2: f32, eps: f32 },
    /// Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
    AdamW { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32 },
    /// Layer-wise Adaptive Moments for Batch training (You et al., 2019).
    /// Burn provides a shader-optimised implementation we delegate to.
    Lamb { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32 },
}

impl OptimizerKind {
    pub fn name(&self) -> &'static str {
        match self {
            OptimizerKind::Sgd   { .. } => "SGD",
            OptimizerKind::Adam  { .. } => "Adam",
            OptimizerKind::AdamW { .. } => "AdamW",
            OptimizerKind::Lamb  { .. } => "LAMB",
        }
    }

    pub fn lr(&self) -> f32 {
        match *self {
            OptimizerKind::Sgd   { lr, .. }
            | OptimizerKind::Adam  { lr, .. }
            | OptimizerKind::AdamW { lr, .. }
            | OptimizerKind::Lamb  { lr, .. } => lr,
        }
    }

    pub fn set_lr(&mut self, new_lr: f32) {
        match self {
            OptimizerKind::Sgd   { lr, .. }
            | OptimizerKind::Adam  { lr, .. }
            | OptimizerKind::AdamW { lr, .. }
            | OptimizerKind::Lamb  { lr, .. } => *lr = new_lr,
        }
    }
}

/// Back-compat alias used by older code paths. Modern training uses Burn
/// optimizers directly via `nn::Model::train_step`, which constructs the
/// optimizer from the `OptimizerKind` each time.
pub type Optimizer = OptimizerKind;

#[cfg(test)]
mod tests {
    use crate::backend::CpuAutodiffBackend;
    use burn::module::{Module, Param};
    use burn::optim::{
        AdamConfig, AdamWConfig, GradientsParams, Optimizer as BurnOptimizer, SgdConfig,
    };
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::Tensor as BurnTensor;

    /// Minimal Burn module wrapping a single parameter so we can verify the
    /// Burn optimizers actually descend on a quadratic.
    #[derive(Module, Debug)]
    struct Scalar<B: burn::tensor::backend::Backend> {
        value: Param<BurnTensor<B, 1>>,
    }

    fn init_scalar<B: burn::tensor::backend::Backend>(device: &B::Device) -> Scalar<B> {
        let v = BurnTensor::<B, 1>::from_data([0.0_f32], device);
        Scalar { value: Param::from_tensor(v) }
    }

    fn quadratic_grad<B: AutodiffBackend>(model: &Scalar<B>) -> GradientsParams {
        // f(x) = (x - 3)²  →  grad = 2 (x - 3)
        let x = model.value.val();
        let target = BurnTensor::<B, 1>::from_data([3.0_f32], &x.device());
        let diff = x - target;
        let loss = diff.clone() * diff;
        let grads = loss.backward();
        GradientsParams::from_grads(grads, model)
    }

    #[test]
    fn sgd_descends_on_simple_quadratic() {
        let device = <CpuAutodiffBackend as burn::tensor::backend::Backend>::Device::default();
        let mut model: Scalar<CpuAutodiffBackend> = init_scalar(&device);
        let mut opt = SgdConfig::new().init();
        for _ in 0..400 {
            let grads = quadratic_grad(&model);
            model = opt.step(0.1, model, grads);
        }
        let x = model.value.val().into_data().to_vec::<f32>().unwrap()[0];
        assert!((x - 3.0).abs() < 1e-2, "x = {x}");
    }

    #[test]
    fn adam_converges_too() {
        let device = <CpuAutodiffBackend as burn::tensor::backend::Backend>::Device::default();
        let mut model: Scalar<CpuAutodiffBackend> = init_scalar(&device);
        let mut opt = AdamConfig::new().init();
        for _ in 0..600 {
            let grads = quadratic_grad(&model);
            model = opt.step(0.1, model, grads);
        }
        let x = model.value.val().into_data().to_vec::<f32>().unwrap()[0];
        assert!((x - 3.0).abs() < 5e-2, "x = {x}");
    }

    #[test]
    fn adamw_converges() {
        let device = <CpuAutodiffBackend as burn::tensor::backend::Backend>::Device::default();
        let mut model: Scalar<CpuAutodiffBackend> = init_scalar(&device);
        let mut opt = AdamWConfig::new().with_weight_decay(0.01).init();
        for _ in 0..600 {
            let grads = quadratic_grad(&model);
            model = opt.step(0.1, model, grads);
        }
        let x = model.value.val().into_data().to_vec::<f32>().unwrap()[0];
        assert!((x - 3.0).abs() < 0.5, "x = {x}");
    }
}
