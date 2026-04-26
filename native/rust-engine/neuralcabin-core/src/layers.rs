/// Pure-compute layer forward passes (no autograd state).
/// The JS wrapper retains the autograd graph; Rust handles the FLOP-heavy math.
use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::cpu;
use crate::dtypes::DType;
use crate::tensor::Tensor;

// ── Linear ────────────────────────────────────────────────────────────────────

/// Linear forward: output = input @ weight.T + bias.
/// input:  [B, in_features]
/// weight: [out_features, in_features]
/// bias:   [out_features] (optional; pass empty slice to skip)
/// Returns [B, out_features].
pub fn linear_forward(
  input: &Tensor,
  weight: &Tensor,
  bias: Option<&Tensor>,
  out_dtype: DType,
) -> Result<Tensor> {
  if input.rank() != 2 {
    bail!("linear_forward: input must be rank-2, got rank {}", input.rank());
  }
  if weight.rank() != 2 {
    bail!("linear_forward: weight must be rank-2, got rank {}", weight.rank());
  }
  let (b, in_f) = (input.shape()[0], input.shape()[1]);
  let (out_f, in_f2) = (weight.shape()[0], weight.shape()[1]);
  if in_f != in_f2 {
    bail!("linear_forward: input dim {} != weight in_features {}", in_f, in_f2);
  }

  // output[i,j] = sum_k input[i,k] * weight[j,k]  (weight is transposed)
  let iv = input.to_f32();
  let wv = weight.to_f32();
  let mut out = vec![0.0f32; b * out_f];

  out.par_chunks_mut(out_f).enumerate().for_each(|(i, out_row)| {
    let in_row = &iv[i * in_f..(i + 1) * in_f];
    for j in 0..out_f {
      let w_row = &wv[j * in_f2..(j + 1) * in_f2];
      let mut s = 0.0f32;
      for k in 0..in_f {
        s += in_row[k] * w_row[k];
      }
      out_row[j] = s;
    }
  });

  if let Some(b_tensor) = bias {
    if b_tensor.len() != out_f {
      bail!("linear_forward: bias length {} != out_features {}", b_tensor.len(), out_f);
    }
    let bv = b_tensor.to_f32();
    out.par_chunks_mut(out_f).for_each(|row| {
      for (j, v) in row.iter_mut().enumerate() {
        *v += bv[j];
      }
    });
  }

  Tensor::from_f32(vec![b, out_f], out_dtype, &out)
}

// ── Embedding layer ───────────────────────────────────────────────────────────

/// Embedding lookup with scatter-add gradient accumulation.
/// weights: [vocab_size, embed_dim]; ids: [batch_size].
/// Returns [batch_size, embed_dim].
pub fn embedding_forward(weights: &Tensor, ids: &[i32], out_dtype: DType) -> Result<Tensor> {
  cpu::embedding(weights, ids, out_dtype)
}

/// Scatter-add: accumulate grad_output into grad_weight.
/// grad_output: [B, D]; ids: [B]; grad_weight: flat [V*D] (mutated in place).
pub fn embedding_backward(grad_output: &[f32], ids: &[i32], vocab_size: usize, embed_dim: usize) -> Vec<f32> {
  let mut grad_weight = vec![0.0f32; vocab_size * embed_dim];
  for (i, &id) in ids.iter().enumerate() {
    let id = id as usize;
    if id >= vocab_size { continue; }
    let src = &grad_output[i * embed_dim..(i + 1) * embed_dim];
    let dst = &mut grad_weight[id * embed_dim..(id + 1) * embed_dim];
    for (d, s) in dst.iter_mut().zip(src.iter()) {
      *d += s;
    }
  }
  grad_weight
}

// ── Sequential helper ─────────────────────────────────────────────────────────

/// Apply a sequence of layer descriptors to an input tensor.
/// Each descriptor is a (kind, params) pair where kind is:
///   "linear"     → params = [out_features, in_features, w0..wN, b0..bM]
///   "relu" / "leaky_relu" / "tanh" / "sigmoid" / "gelu" → activation (no params)
/// This is a thin inference-only helper; training uses the JS autograd graph.
pub fn sequential_forward_inference(
  input: &[f32],
  input_shape: &[usize],
  layers: &[LayerDesc],
  out_dtype: DType,
) -> Result<Vec<f32>> {
  let mut current =
    Tensor::from_f32(input_shape.to_vec(), DType::F32, input)?;

  for layer in layers {
    current = match layer {
      LayerDesc::Linear { weight, bias, in_f, out_f } => {
        let w = Tensor::from_f32(vec![*out_f, *in_f], DType::F32, weight)?;
        let b = if bias.is_empty() {
          None
        } else {
          Some(Tensor::from_f32(vec![*out_f], DType::F32, bias)?)
        };
        linear_forward(&current, &w, b.as_ref(), out_dtype)?
      }
      LayerDesc::Relu => cpu::relu(&current, out_dtype)?,
      LayerDesc::LeakyRelu { alpha } => cpu::leaky_relu(&current, *alpha, out_dtype)?,
      LayerDesc::Tanh => cpu::tanh_el(&current, out_dtype)?,
      LayerDesc::Sigmoid => cpu::sigmoid_el(&current, out_dtype)?,
      LayerDesc::Gelu => {
        let (out, _tcache) = cpu::gelu_el(&current, out_dtype)?;
        out
      }
      LayerDesc::Softmax => cpu::softmax(&current, out_dtype)?,
    };
  }

  Ok(current.to_f32())
}

#[derive(Debug, Clone)]
pub enum LayerDesc {
  Linear { in_f: usize, out_f: usize, weight: Vec<f32>, bias: Vec<f32> },
  Relu,
  LeakyRelu { alpha: f32 },
  Tanh,
  Sigmoid,
  Gelu,
  Softmax,
}
