use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::dtypes::DType;
use crate::tensor::Tensor;

// ── AXPY kernel (AVX2 fused-multiply-add when available) ─────────────────────
#[inline(always)]
fn axpy(out: &mut [f32], src: &[f32], alpha: f32) {
  debug_assert_eq!(out.len(), src.len());
  #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
  unsafe {
    use core::arch::x86_64::*;
    let a = _mm256_set1_ps(alpha);
    let mut i = 0usize;
    while i + 8 <= out.len() {
      let y = _mm256_loadu_ps(out.as_ptr().add(i));
      let x = _mm256_loadu_ps(src.as_ptr().add(i));
      let z = _mm256_fmadd_ps(a, x, y);
      _mm256_storeu_ps(out.as_mut_ptr().add(i), z);
      i += 8;
    }
    while i < out.len() {
      out[i] += alpha * src[i];
      i += 1;
    }
    return;
  }
  for (o, s) in out.iter_mut().zip(src.iter()) {
    *o += alpha * *s;
  }
}

// ── Mulberry32 PRNG (matches JS rngFromSeed) ─────────────────────────────────
#[inline]
fn mulberry32(s: &mut u32) -> f32 {
  *s = s.wrapping_add(0x6D2B79F5);
  let mut t = *s;
  t = (t ^ (t >> 15)).wrapping_mul(t | 1);
  t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
  ((t ^ (t >> 14)) as u64 & 0xFFFF_FFFF) as f32 / 4_294_967_296.0
}

// ── Element-wise ops ──────────────────────────────────────────────────────────

pub fn add(a: &Tensor, b: &Tensor, out_dtype: DType) -> Result<Tensor> {
  if !a.same_shape(b) {
    // Bias-broadcast: a is [B, N], b is [N]
    if a.rank() == 2 && b.rank() == 1 && a.shape()[1] == b.shape()[0] {
      let (bsz, n) = (a.shape()[0], a.shape()[1]);
      let av = a.to_f32();
      let bv = b.to_f32();
      let out: Vec<f32> = av
        .par_chunks(n)
        .flat_map(|row| row.iter().enumerate().map(|(j, &x)| x + bv[j]).collect::<Vec<_>>())
        .collect();
      return Tensor::from_f32(vec![bsz, n], out_dtype, &out);
    }
    bail!("add: incompatible shapes {:?} and {:?}", a.shape(), b.shape());
  }
  let av = a.to_f32();
  let bv = b.to_f32();
  let out: Vec<f32> = av.par_iter().zip(bv.par_iter()).map(|(x, y)| x + y).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

pub fn sub(a: &Tensor, b: &Tensor, out_dtype: DType) -> Result<Tensor> {
  if !a.same_shape(b) {
    bail!("sub: shape mismatch {:?} vs {:?}", a.shape(), b.shape());
  }
  let av = a.to_f32();
  let bv = b.to_f32();
  let out: Vec<f32> = av.par_iter().zip(bv.par_iter()).map(|(x, y)| x - y).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

pub fn mul(a: &Tensor, b: &Tensor, out_dtype: DType) -> Result<Tensor> {
  if !a.same_shape(b) {
    bail!("mul expects same shape: {:?} vs {:?}", a.shape(), b.shape());
  }
  let av = a.to_f32();
  let bv = b.to_f32();
  let out: Vec<f32> = av.par_iter().zip(bv.par_iter()).map(|(x, y)| x * y).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

pub fn mul_scalar(a: &Tensor, s: f32, out_dtype: DType) -> Result<Tensor> {
  let av = a.to_f32();
  let out: Vec<f32> = av.par_iter().map(|x| x * s).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

// ── Activations ───────────────────────────────────────────────────────────────

pub fn relu(a: &Tensor, out_dtype: DType) -> Result<Tensor> {
  let av = a.to_f32();
  let out: Vec<f32> = av.par_iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

pub fn leaky_relu(a: &Tensor, alpha: f32, out_dtype: DType) -> Result<Tensor> {
  let av = a.to_f32();
  let out: Vec<f32> = av.par_iter().map(|v| if *v > 0.0 { *v } else { v * alpha }).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

pub fn tanh_el(a: &Tensor, out_dtype: DType) -> Result<Tensor> {
  let av = a.to_f32();
  let out: Vec<f32> = av.par_iter().map(|v| v.tanh()).collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

pub fn sigmoid_el(a: &Tensor, out_dtype: DType) -> Result<Tensor> {
  let av = a.to_f32();
  let out: Vec<f32> = av
    .par_iter()
    .map(|&x| if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) })
    .collect();
  Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)
}

/// GELU (tanh approximation). Returns (output, tcache) where tcache holds the
/// inner tanh values needed for the backward pass.
pub fn gelu_el(a: &Tensor, out_dtype: DType) -> Result<(Tensor, Vec<f32>)> {
  let av = a.to_f32();
  let c = (2.0f32 / std::f32::consts::PI).sqrt();
  let (out, tcache): (Vec<f32>, Vec<f32>) = av
    .par_iter()
    .map(|&x| {
      let t = (c * (x + 0.044715 * x * x * x)).tanh();
      (0.5 * x * (1.0 + t), t)
    })
    .unzip();
  Ok((Tensor::from_f32(a.shape().to_vec(), out_dtype, &out)?, tcache))
}

// ── Softmax ───────────────────────────────────────────────────────────────────

/// Row-wise softmax on a 2D tensor [B, N]. Returns the probability tensor.
pub fn softmax(a: &Tensor, out_dtype: DType) -> Result<Tensor> {
  if a.rank() != 2 {
    bail!("softmax requires rank-2 tensor, got rank {}", a.rank());
  }
  let (b, n) = (a.shape()[0], a.shape()[1]);
  let av = a.to_f32();
  let mut out = vec![0.0f32; b * n];

  out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
    let in_row = &av[i * n..(i + 1) * n];
    let maxv = in_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
      *o = (x - maxv).exp();
      sum += *o;
    }
    let inv = 1.0 / sum;
    for o in out_row.iter_mut() {
      *o *= inv;
    }
  });

  Tensor::from_f32(vec![b, n], out_dtype, &out)
}

// ── Losses ────────────────────────────────────────────────────────────────────

/// Fused softmax + cross-entropy. Returns (scalar loss, probs_cache).
/// probs_cache is needed by the caller's backward closure.
pub fn softmax_cross_entropy(
  logits: &Tensor,
  labels: &[i32],
  out_dtype: DType,
) -> Result<(f32, Vec<f32>)> {
  if logits.rank() != 2 {
    bail!("softmax_cross_entropy requires rank-2 logits");
  }
  let (b, c) = (logits.shape()[0], logits.shape()[1]);
  if labels.len() != b {
    bail!("labels length {} != batch size {}", labels.len(), b);
  }
  let lv = logits.to_f32();
  let mut probs = vec![0.0f32; b * c];
  let mut loss = 0.0f32;

  for i in 0..b {
    let row = i * c;
    let in_row = &lv[row..row + c];
    let maxv = in_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (j, &x) in in_row.iter().enumerate() {
      let e = (x - maxv).exp();
      probs[row + j] = e;
      sum += e;
    }
    let inv = 1.0 / sum;
    for p in &mut probs[row..row + c] {
      *p *= inv;
    }
    let y = labels[i] as usize;
    if y >= c {
      // Skip out-of-range labels rather than panicking in a hot path.
      continue;
    }
    let py = probs[row + y].max(1e-12);
    loss += -py.ln();
  }

  Ok((loss / b as f32, probs))
}

/// MSE loss: mean((a-b)^2). Returns scalar loss value.
pub fn mse_loss(a: &Tensor, b: &Tensor) -> Result<f32> {
  if !a.same_shape(b) {
    bail!("mse_loss: shape mismatch {:?} vs {:?}", a.shape(), b.shape());
  }
  let av = a.to_f32();
  let bv = b.to_f32();
  let sum: f32 = av
    .par_iter()
    .zip(bv.par_iter())
    .map(|(x, y)| { let d = x - y; d * d })
    .sum();
  Ok(sum / av.len() as f32)
}

// ── Dropout ───────────────────────────────────────────────────────────────────

/// Inverted dropout. Returns (output, mask). Mask is 0 or scale=1/(1-p).
pub fn dropout(a: &Tensor, p: f32, seed: u32, out_dtype: DType) -> Result<(Tensor, Vec<f32>)> {
  if p <= 0.0 {
    let out = Tensor::from_f32(a.shape().to_vec(), out_dtype, &a.to_f32())?;
    let mask = vec![1.0f32; a.len()];
    return Ok((out, mask));
  }
  let av = a.to_f32();
  let scale = 1.0 / (1.0 - p);
  let mut rng_state = seed;
  let mut out_data = av.clone();
  let mut mask = vec![scale; av.len()];
  for (i, (o, m)) in out_data.iter_mut().zip(mask.iter_mut()).enumerate() {
    // Advance RNG per element with a unique offset so different elements get
    // independent draws even when called with the same seed.
    let mut s = rng_state.wrapping_add(i as u32 * 2654435761u32);
    if mulberry32(&mut s) < p {
      *o = 0.0;
      *m = 0.0;
    }
  }
  // Advance master state for caller reproducibility
  rng_state = rng_state.wrapping_add(av.len() as u32);
  let _ = rng_state;
  let out = Tensor::from_f32(a.shape().to_vec(), out_dtype, &out_data)?;
  Ok((out, mask))
}

// ── Embedding ─────────────────────────────────────────────────────────────────

/// Embedding lookup: weights [V, D], ids[] length B → [B, D].
pub fn embedding(weights: &Tensor, ids: &[i32], out_dtype: DType) -> Result<Tensor> {
  if weights.rank() != 2 {
    bail!("embedding requires rank-2 weight tensor");
  }
  let (v, d) = (weights.shape()[0], weights.shape()[1]);
  let b = ids.len();
  let wv = weights.to_f32();
  let mut out = vec![0.0f32; b * d];

  out.par_chunks_mut(d).enumerate().for_each(|(i, out_row)| {
    let id = ids[i] as usize;
    if id < v {
      out_row.copy_from_slice(&wv[id * d..(id + 1) * d]);
    }
  });

  Tensor::from_f32(vec![b, d], out_dtype, &out)
}

// ── Reduction ops ─────────────────────────────────────────────────────────────

pub fn sum_all(a: &Tensor) -> f32 {
  a.to_f32().par_iter().sum()
}

// ── Random tensor generation ──────────────────────────────────────────────────

/// Box-Muller normal distribution using Mulberry32 PRNG (matches JS rngFromSeed).
pub fn randn(shape: Vec<usize>, seed: u32, out_dtype: DType) -> Result<Tensor> {
  let size: usize = shape.iter().product();
  let mut data = vec![0.0f32; size];
  let mut s = seed;
  let mut i = 0usize;
  while i < size {
    let u = loop { let v = mulberry32(&mut s); if v != 0.0 { break v; } };
    let v = loop { let v = mulberry32(&mut s); if v != 0.0 { break v; } };
    let mag = (-2.0 * u.ln()).sqrt();
    data[i] = mag * (2.0 * std::f32::consts::PI * v).cos();
    if i + 1 < size {
      data[i + 1] = mag * (2.0 * std::f32::consts::PI * v).sin();
    }
    i += 2;
  }
  Tensor::from_f32(shape, out_dtype, &data)
}

// ── Matrix multiply ───────────────────────────────────────────────────────────

pub fn matmul(a: &Tensor, b: &Tensor, out_dtype: DType) -> Result<Tensor> {
  if a.rank() != 2 || b.rank() != 2 {
    bail!("matmul requires rank-2 tensors");
  }
  let (m, k) = (a.shape()[0], a.shape()[1]);
  let (k2, n) = (b.shape()[0], b.shape()[1]);
  if k != k2 {
    bail!("matmul shape mismatch: {:?} x {:?}", a.shape(), b.shape());
  }

  let av = a.to_f32();
  let bv = b.to_f32();
  let mut out = vec![0.0f32; m * n];

  out.par_chunks_mut(n).enumerate().for_each(|(row_idx, out_row)| {
    let a_row = &av[row_idx * k..(row_idx + 1) * k];
    for kk in 0..k {
      let alpha = a_row[kk];
      let b_row = &bv[kk * n..(kk + 1) * n];
      axpy(out_row, b_row, alpha);
    }
  });

  Tensor::from_f32(vec![m, n], out_dtype, &out)
}

// ── NaN / Inf guard ───────────────────────────────────────────────────────────

/// Returns true if tensor contains any NaN or infinite value.
pub fn has_nan_or_inf(a: &Tensor) -> bool {
  a.to_f32().par_iter().any(|v| v.is_nan() || v.is_infinite())
}
