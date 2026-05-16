//! Loss functions — eager (CPU) for evaluation and Burn-graph variants for
//! training.

use crate::activations::softmax_rows;
use crate::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::activation::log_softmax;
use burn::tensor::Tensor as BurnTensor;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Loss {
    /// Mean-squared error. Pair with `Activation::Identity`, `Sigmoid`, or `Tanh`.
    MeanSquaredError,
    /// Softmax + categorical cross-entropy. The model's last layer should be
    /// `Activation::Identity` — softmax is applied internally for numerical
    /// stability.
    CrossEntropy,
}

impl Loss {
    pub fn name(&self) -> &'static str {
        match self {
            Loss::MeanSquaredError => "MeanSquaredError",
            Loss::CrossEntropy => "CrossEntropy",
        }
    }

    pub fn all() -> &'static [Loss] {
        &[Loss::MeanSquaredError, Loss::CrossEntropy]
    }

    /// Compute the loss inside Burn's autodiff graph. Returns a scalar tensor.
    ///
    /// - MSE        : mean over all elements of (pred - target)²
    /// - CrossEntropy: softmax + categorical cross-entropy over the class axis.
    ///   `target` is expected to be one-hot encoded with shape `(batch, classes)`.
    pub fn forward_burn<B: Backend>(
        &self,
        pred: BurnTensor<B, 2>,
        target: BurnTensor<B, 2>,
    ) -> BurnTensor<B, 1> {
        match self {
            Loss::MeanSquaredError => {
                let diff = pred - target;
                let sq = diff.clone() * diff;
                sq.mean()
            }
            Loss::CrossEntropy => {
                // log_softmax + categorical cross-entropy, averaged across batch.
                // log_softmax expects the class dim; for `(batch, classes)` that's dim=1.
                let log_p = log_softmax(pred, 1);
                -(target * log_p).sum_dim(1).mean()
            }
        }
    }

    /// Eager scalar — handy for evaluating a model without a Burn graph.
    pub fn eval(&self, output: &Tensor, target: &Tensor) -> f32 {
        match self {
            Loss::MeanSquaredError => {
                assert_eq!(output.shape, target.shape, "MSE shape mismatch");
                let n = output.data.len() as f32;
                let mut s = 0.0_f32;
                for (a, b) in output.data.iter().zip(&target.data) {
                    let d = a - b;
                    s += d * d;
                }
                s / n
            }
            Loss::CrossEntropy => {
                assert_eq!(output.shape, target.shape, "CE shape mismatch");
                let sm = softmax_rows(output);
                let rows = sm.rows() as f32;
                let mut s = 0.0_f32;
                for (p, t) in sm.data.iter().zip(&target.data) {
                    s -= t * p.max(1e-12).ln();
                }
                s / rows
            }
        }
    }
}
