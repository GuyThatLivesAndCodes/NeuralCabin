//! Activation functions.
//!
//! Activations are stored as a serializable enum so models can be persisted.
//! Two surfaces exist:
//! - `apply(&Tensor)` — pure CPU op used for inference, eager evaluation, and
//!   serialization round-trips.
//! - `apply_burn(...)` — applies the activation to a Burn tensor inside the
//!   training loop (called from `nn::Model::train_step`).

use crate::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::activation;
use burn::tensor::Tensor as BurnTensor;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    /// Row-wise softmax. Use only as the last layer with `Loss::CrossEntropy`,
    /// or as a pure inference op.
    Softmax,
}

impl Activation {
    pub fn name(&self) -> &'static str {
        match self {
            Activation::Identity => "Identity",
            Activation::ReLU => "ReLU",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::Softmax => "Softmax",
        }
    }

    pub fn all() -> &'static [Activation] {
        &[
            Activation::Identity,
            Activation::ReLU,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax,
        ]
    }

    /// Pure-tensor forward (CPU). Used for inference and inside the
    /// softmax-cross-entropy loss.
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Identity => x.clone(),
            Activation::ReLU => x.map(|v| if v > 0.0 { v } else { 0.0 }),
            Activation::Sigmoid => x.map(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.map(|v| v.tanh()),
            Activation::Softmax => softmax_rows(x),
        }
    }

    /// Apply this activation inside Burn's computation graph.
    ///
    /// Softmax during training is normally folded into the cross-entropy loss
    /// (`loss::Loss::CrossEntropy` does this), so this path returns the raw
    /// logits unchanged for `Softmax` — the loss handles it.
    pub fn apply_burn<B: Backend>(&self, x: BurnTensor<B, 2>) -> BurnTensor<B, 2> {
        match self {
            Activation::Identity => x,
            Activation::ReLU => activation::relu(x),
            Activation::Sigmoid => activation::sigmoid(x),
            Activation::Tanh => x.tanh(),
            // The loss layer applies the softmax internally for numerical
            // stability (`log_softmax + nll`). Returning the logits here is
            // correct for that combo.
            Activation::Softmax => x,
        }
    }
}

/// Numerically-stable row-wise softmax for a 2-D tensor.
pub fn softmax_rows(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "softmax_rows expects (batch, classes)");
    let (rows, cols) = (x.rows(), x.cols());
    let mut out = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let row = &x.data[i * cols..(i + 1) * cols];
        let mut m = f32::NEG_INFINITY;
        for &v in row { if v > m { m = v; } }
        let mut sum = 0.0_f32;
        for j in 0..cols {
            let e = (row[j] - m).exp();
            out[i * cols + j] = e;
            sum += e;
        }
        for j in 0..cols { out[i * cols + j] /= sum; }
    }
    Tensor::new(vec![rows, cols], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_sums_to_one() {
        let x = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
        let s = softmax_rows(&x);
        for i in 0..2 {
            let sum: f32 = s.data[i * 3..(i + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
