//! Optimisers: parameter update rules.
//!
//! Implements SGD (with optional momentum), Adam, AdamW, and LAMB — all from
//! scratch with no external dependencies.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OptimizerKind {
    Sgd { lr: f32, momentum: f32 },
    Adam { lr: f32, beta1: f32, beta2: f32, eps: f32 },
    /// Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
    /// Weight decay is applied directly to the parameter, not to the gradient.
    AdamW { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32 },
    /// Layer-wise Adaptive Moments for Batch training (You et al., 2019).
    /// Scales the update by the ratio of the parameter norm to the update norm,
    /// enabling stable training with very large batch sizes.
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

/// Per-parameter optimiser state.
pub struct Optimizer {
    pub kind: OptimizerKind,
    pub step: u64,
    pub momentum: Vec<Tensor>,   // SGD momentum / Adam m buffers
    pub variance: Vec<Tensor>,   // Adam v buffers
}

impl Optimizer {
    pub fn new(kind: OptimizerKind, param_shapes: &[Vec<usize>]) -> Self {
        let momentum = param_shapes.iter().map(|s| Tensor::zeros(s.clone())).collect();
        let variance = param_shapes.iter().map(|s| Tensor::zeros(s.clone())).collect();
        Self { kind, step: 0, momentum, variance }
    }

    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &[Option<Tensor>]) {
        self.step += 1;
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.momentum.len());

        match self.kind {
            // ── SGD ──────────────────────────────────────────────────────────
            OptimizerKind::Sgd { lr, momentum } => {
                for (i, p) in params.iter_mut().enumerate() {
                    let g = match &grads[i] { Some(g) => g, None => continue };
                    if momentum != 0.0 {
                        let m = &mut self.momentum[i];
                        for (mv, gv) in m.data.iter_mut().zip(&g.data) {
                            *mv = momentum * *mv + *gv;
                        }
                        p.axpy_inplace(-lr, m).expect("sgd step");
                    } else {
                        p.axpy_inplace(-lr, g).expect("sgd step");
                    }
                }
            }

            // ── Adam ─────────────────────────────────────────────────────────
            OptimizerKind::Adam { lr, beta1, beta2, eps } => {
                let t   = self.step as f32;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);
                for (i, p) in params.iter_mut().enumerate() {
                    let g = match &grads[i] { Some(g) => g, None => continue };
                    let m = &mut self.momentum[i];
                    let v = &mut self.variance[i];
                    for ((mv, vv), gv) in m.data.iter_mut().zip(v.data.iter_mut()).zip(&g.data) {
                        *mv = beta1 * *mv + (1.0 - beta1) * *gv;
                        *vv = beta2 * *vv + (1.0 - beta2) * (*gv) * (*gv);
                    }
                    for (j, pv) in p.data.iter_mut().enumerate() {
                        let mhat = m.data[j] / bc1;
                        let vhat = v.data[j] / bc2;
                        *pv -= lr * mhat / (vhat.sqrt() + eps);
                    }
                }
            }

            // ── AdamW ────────────────────────────────────────────────────────
            // w_{t+1} = w_t − lr * (m̂/(√v̂+ε) + λ·w_t)
            OptimizerKind::AdamW { lr, beta1, beta2, eps, weight_decay } => {
                let t   = self.step as f32;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);
                for (i, p) in params.iter_mut().enumerate() {
                    let g = match &grads[i] { Some(g) => g, None => continue };
                    let m = &mut self.momentum[i];
                    let v = &mut self.variance[i];
                    for ((mv, vv), gv) in m.data.iter_mut().zip(v.data.iter_mut()).zip(&g.data) {
                        *mv = beta1 * *mv + (1.0 - beta1) * *gv;
                        *vv = beta2 * *vv + (1.0 - beta2) * (*gv) * (*gv);
                    }
                    for (j, pv) in p.data.iter_mut().enumerate() {
                        let mhat = m.data[j] / bc1;
                        let vhat = v.data[j] / bc2;
                        // Decoupled weight decay applied to the parameter directly.
                        *pv -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * *pv);
                    }
                }
            }

            // ── LAMB ─────────────────────────────────────────────────────────
            // r_t = m̂/(√v̂+ε) + λ·w_t
            // trust = ‖w‖ / ‖r‖  (clamped so neither norm is 0)
            // w_{t+1} = w_t − lr · trust · r_t
            OptimizerKind::Lamb { lr, beta1, beta2, eps, weight_decay } => {
                let t   = self.step as f32;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);
                for (i, p) in params.iter_mut().enumerate() {
                    let g = match &grads[i] { Some(g) => g, None => continue };
                    let m = &mut self.momentum[i];
                    let v = &mut self.variance[i];
                    // Update moments.
                    for ((mv, vv), gv) in m.data.iter_mut().zip(v.data.iter_mut()).zip(&g.data) {
                        *mv = beta1 * *mv + (1.0 - beta1) * *gv;
                        *vv = beta2 * *vv + (1.0 - beta2) * (*gv) * (*gv);
                    }
                    // Compute per-element update ratio r.
                    let r: Vec<f32> = (0..p.data.len()).map(|j| {
                        let mhat = m.data[j] / bc1;
                        let vhat = v.data[j] / bc2;
                        mhat / (vhat.sqrt() + eps) + weight_decay * p.data[j]
                    }).collect();
                    // Layer-wise trust ratio.
                    let w_norm: f32 = p.data.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let r_norm: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let trust = if w_norm > 0.0 && r_norm > 0.0 { w_norm / r_norm } else { 1.0 };
                    for (j, pv) in p.data.iter_mut().enumerate() {
                        *pv -= lr * trust * r[j];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_descends_on_simple_quadratic() {
        let shapes = vec![vec![1]];
        let mut opt = Optimizer::new(OptimizerKind::Sgd { lr: 0.1, momentum: 0.0 }, &shapes);
        let mut x = Tensor::new(vec![1], vec![0.0]);
        for _ in 0..200 {
            let g = Tensor::new(vec![1], vec![2.0 * (x.data[0] - 3.0)]);
            opt.step(&mut [&mut x], &[Some(g)]);
        }
        assert!((x.data[0] - 3.0).abs() < 1e-3, "x = {}", x.data[0]);
    }

    #[test]
    fn adam_converges_too() {
        let shapes = vec![vec![1]];
        let mut opt = Optimizer::new(
            OptimizerKind::Adam { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
            &shapes,
        );
        let mut x = Tensor::new(vec![1], vec![0.0]);
        for _ in 0..500 {
            let g = Tensor::new(vec![1], vec![2.0 * (x.data[0] - 3.0)]);
            opt.step(&mut [&mut x], &[Some(g)]);
        }
        assert!((x.data[0] - 3.0).abs() < 5e-2, "x = {}", x.data[0]);
    }

    #[test]
    fn adamw_converges() {
        let shapes = vec![vec![1]];
        let mut opt = Optimizer::new(
            OptimizerKind::AdamW { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 },
            &shapes,
        );
        let mut x = Tensor::new(vec![1], vec![0.0]);
        for _ in 0..500 {
            let g = Tensor::new(vec![1], vec![2.0 * (x.data[0] - 3.0)]);
            opt.step(&mut [&mut x], &[Some(g)]);
        }
        // With weight decay the equilibrium shifts slightly but should still be close.
        assert!((x.data[0] - 3.0).abs() < 0.5, "x = {}", x.data[0]);
    }

    #[test]
    fn lamb_converges() {
        let shapes = vec![vec![1]];
        let mut opt = Optimizer::new(
            OptimizerKind::Lamb { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 },
            &shapes,
        );
        let mut x = Tensor::new(vec![1], vec![0.0]);
        for _ in 0..500 {
            let g = Tensor::new(vec![1], vec![2.0 * (x.data[0] - 3.0)]);
            opt.step(&mut [&mut x], &[Some(g)]);
        }
        assert!((x.data[0] - 3.0).abs() < 0.5, "x = {}", x.data[0]);
    }
}
