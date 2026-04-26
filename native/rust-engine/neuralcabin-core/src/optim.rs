/// Optimizer step functions — pure arithmetic, no autograd.
/// State vectors (m, v, momentum) are owned by the JS side and passed in / out
/// each step so that serialization and model-cache lifetime remain in JS.
use rayon::prelude::*;

// ── SGD with momentum ─────────────────────────────────────────────────────────

/// One SGD step.
/// Returns updated (params, momentum_buf).
/// weight_decay is L2; set to 0 to disable.
pub fn sgd_step(
  params: &[f32],
  grads: &[f32],
  buf: &[f32],  // momentum buffer (same len as params; zeros if first step)
  lr: f32,
  momentum: f32,
  weight_decay: f32,
) -> (Vec<f32>, Vec<f32>) {
  assert_eq!(params.len(), grads.len());
  assert_eq!(params.len(), buf.len());

  let new_buf: Vec<f32> = if momentum > 0.0 {
    buf
      .par_iter()
      .zip(grads.par_iter())
      .zip(params.par_iter())
      .map(|((&b, &g), &p)| {
        let g_wd = g + weight_decay * p;
        momentum * b + g_wd
      })
      .collect()
  } else {
    grads
      .par_iter()
      .zip(params.par_iter())
      .map(|(&g, &p)| g + weight_decay * p)
      .collect()
  };

  let new_params: Vec<f32> = params
    .par_iter()
    .zip(new_buf.par_iter())
    .map(|(&p, &nb)| p - lr * nb)
    .collect();

  (new_params, new_buf)
}

// ── Adam ──────────────────────────────────────────────────────────────────────

/// One Adam step.
/// Returns updated (params, m, v).
/// `t` is the 1-based step counter (call with 1, 2, 3, …).
pub fn adam_step(
  params: &[f32],
  grads: &[f32],
  m: &[f32],
  v: &[f32],
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  t: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  assert_eq!(params.len(), grads.len());
  assert_eq!(params.len(), m.len());
  assert_eq!(params.len(), v.len());

  let bc1 = 1.0 - beta1.powi(t as i32);
  let bc2 = 1.0 - beta2.powi(t as i32);
  let alpha = lr * bc2.sqrt() / bc1;

  let (new_m, new_v): (Vec<f32>, Vec<f32>) = grads
    .par_iter()
    .zip(m.par_iter())
    .zip(v.par_iter())
    .map(|((&g, &mi), &vi)| (beta1 * mi + (1.0 - beta1) * g, beta2 * vi + (1.0 - beta2) * g * g))
    .unzip();

  let new_params: Vec<f32> = params
    .par_iter()
    .zip(new_m.par_iter())
    .zip(new_v.par_iter())
    .map(|((&p, &mi), &vi)| p - alpha * mi / (vi.sqrt() + eps))
    .collect();

  (new_params, new_m, new_v)
}

// ── AdamW ─────────────────────────────────────────────────────────────────────

/// One AdamW step (decoupled weight decay).
/// Returns updated (params, m, v).
pub fn adamw_step(
  params: &[f32],
  grads: &[f32],
  m: &[f32],
  v: &[f32],
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  weight_decay: f32,
  t: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  assert_eq!(params.len(), grads.len());
  assert_eq!(params.len(), m.len());
  assert_eq!(params.len(), v.len());

  let bc1 = 1.0 - beta1.powi(t as i32);
  let bc2 = 1.0 - beta2.powi(t as i32);
  let alpha = lr * bc2.sqrt() / bc1;

  let (new_m, new_v): (Vec<f32>, Vec<f32>) = grads
    .par_iter()
    .zip(m.par_iter())
    .zip(v.par_iter())
    .map(|((&g, &mi), &vi)| (beta1 * mi + (1.0 - beta1) * g, beta2 * vi + (1.0 - beta2) * g * g))
    .unzip();

  let new_params: Vec<f32> = params
    .par_iter()
    .zip(new_m.par_iter())
    .zip(new_v.par_iter())
    .map(|((&p, &mi), &vi)| {
      let adam_update = alpha * mi / (vi.sqrt() + eps);
      p - adam_update - lr * weight_decay * p
    })
    .collect();

  (new_params, new_m, new_v)
}

// ── Gradient clipping ─────────────────────────────────────────────────────────

/// Clip gradients by global L2 norm. Returns clipped gradients.
pub fn clip_grad_norm(grads: &[f32], max_norm: f32) -> Vec<f32> {
  let norm: f32 = grads.par_iter().map(|g| g * g).sum::<f32>().sqrt();
  if norm <= max_norm || norm == 0.0 {
    return grads.to_vec();
  }
  let scale = max_norm / norm;
  grads.par_iter().map(|g| g * scale).collect()
}
