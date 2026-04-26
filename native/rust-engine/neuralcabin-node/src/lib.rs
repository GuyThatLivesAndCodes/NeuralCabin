use std::sync::OnceLock;

use napi::bindgen_prelude::Error;
use napi_derive::napi;
use neuralcabin_core::{
  cpu, layers, neuroevolution, optim, rl, BackendMode, Engine, EngineConfig, PrecisionMode,
};
use parking_lot::RwLock;

// ── Global engine singleton ───────────────────────────────────────────────────

fn global_engine() -> &'static RwLock<Engine> {
  static ENGINE: OnceLock<RwLock<Engine>> = OnceLock::new();
  ENGINE.get_or_init(|| RwLock::new(Engine::new(EngineConfig::default())))
}

fn parse_backend(mode: Option<String>) -> BackendMode {
  match mode.as_deref().unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
    "cpu" => BackendMode::Cpu,
    "gpu" => BackendMode::Gpu,
    _ => BackendMode::Auto,
  }
}

fn parse_precision(mode: Option<String>) -> PrecisionMode {
  match mode.as_deref().unwrap_or("fp32").trim().to_ascii_lowercase().as_str() {
    "fp16" => PrecisionMode::FP16,
    "bf16" => PrecisionMode::BF16,
    _ => PrecisionMode::FP32,
  }
}

fn napi_err(e: impl std::fmt::Display) -> Error {
  Error::from_reason(e.to_string())
}

fn dtype() -> neuralcabin_core::dtypes::DType {
  global_engine().read().config.precision.as_dtype()
}

// ── Backend control ───────────────────────────────────────────────────────────

#[napi(object)]
pub struct BackendInfo {
  pub backend: String,
  pub precision: String,
  pub mixed_precision: bool,
}

#[napi]
pub fn init_backend(mode: Option<String>, precision: Option<String>) {
  let cfg = EngineConfig {
    backend: parse_backend(mode),
    precision: parse_precision(precision),
    operator_fusion: true,
  };
  *global_engine().write() = Engine::new(cfg);
}

#[napi]
pub fn backend_info() -> BackendInfo {
  let e = global_engine().read();
  BackendInfo {
    backend: e.backend_name().to_string(),
    precision: format!("{:?}", e.config.precision),
    mixed_precision: e.supports_mixed_precision(),
  }
}

// ── Tensor ops ────────────────────────────────────────────────────────────────

#[napi]
pub fn matmul(
  a: Vec<f64>, a_rows: u32, a_cols: u32,
  b: Vec<f64>, b_rows: u32, b_cols: u32,
) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(vec![a_rows as usize, a_cols as usize], &av).map_err(napi_err)?;
  let bt = e.tensor_from_f32(vec![b_rows as usize, b_cols as usize], &bv).map_err(napi_err)?;
  let out = e.matmul(&at, &bt).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn add_op(
  a: Vec<f64>, a_rows: u32, a_cols: u32,
  b: Vec<f64>, b_rows: u32, b_cols: u32,
) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let dt = dtype();
  let at = e.tensor_from_f32(vec![a_rows as usize, a_cols as usize], &av).map_err(napi_err)?;
  let bt = e.tensor_from_f32(vec![b_rows as usize, b_cols as usize], &bv).map_err(napi_err)?;
  let out = cpu::add(&at, &bt, dt).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn sub_op(a: Vec<f64>, b: Vec<f64>, rows: u32, cols: u32) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let dt = dtype();
  let at = e.tensor_from_f32(vec![rows as usize, cols as usize], &av).map_err(napi_err)?;
  let bt = e.tensor_from_f32(vec![rows as usize, cols as usize], &bv).map_err(napi_err)?;
  let out = cpu::sub(&at, &bt, dt).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn mul_op(a: Vec<f64>, b: Vec<f64>, rows: u32, cols: u32) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let dt = dtype();
  let at = e.tensor_from_f32(vec![rows as usize, cols as usize], &av).map_err(napi_err)?;
  let bt = e.tensor_from_f32(vec![rows as usize, cols as usize], &bv).map_err(napi_err)?;
  let out = cpu::mul(&at, &bt, dt).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn relu_op(a: Vec<f64>, shape: Vec<u32>) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  let out = cpu::relu(&at, dtype()).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn leaky_relu_op(a: Vec<f64>, shape: Vec<u32>, alpha: f64) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  let out = cpu::leaky_relu(&at, alpha as f32, dtype()).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn tanh_op(a: Vec<f64>, shape: Vec<u32>) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  let out = cpu::tanh_el(&at, dtype()).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn sigmoid_op(a: Vec<f64>, shape: Vec<u32>) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  let out = cpu::sigmoid_el(&at, dtype()).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi(object)]
pub struct GeluResult {
  pub output: Vec<f64>,
  pub tcache: Vec<f64>,
}

#[napi]
pub fn gelu_op(a: Vec<f64>, shape: Vec<u32>) -> napi::Result<GeluResult> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  let (out, tcache) = cpu::gelu_el(&at, dtype()).map_err(napi_err)?;
  Ok(GeluResult {
    output: out.to_f32().iter().map(|&x| x as f64).collect(),
    tcache: tcache.iter().map(|&x| x as f64).collect(),
  })
}

#[napi]
pub fn softmax_op(a: Vec<f64>, rows: u32, cols: u32) -> napi::Result<Vec<f64>> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(vec![rows as usize, cols as usize], &av).map_err(napi_err)?;
  let out = cpu::softmax(&at, dtype()).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi(object)]
pub struct CrossEntropyResult {
  pub loss: f64,
  pub probs: Vec<f64>,
}

#[napi]
pub fn softmax_cross_entropy_op(
  logits: Vec<f64>,
  labels: Vec<i32>,
  rows: u32,
  cols: u32,
) -> napi::Result<CrossEntropyResult> {
  let lv: Vec<f32> = logits.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let lt = e.tensor_from_f32(vec![rows as usize, cols as usize], &lv).map_err(napi_err)?;
  let (loss, probs) = cpu::softmax_cross_entropy(&lt, &labels, dtype()).map_err(napi_err)?;
  Ok(CrossEntropyResult {
    loss: loss as f64,
    probs: probs.iter().map(|&x| x as f64).collect(),
  })
}

#[napi]
pub fn mse_loss_op(a: Vec<f64>, b: Vec<f64>, shape: Vec<u32>) -> napi::Result<f64> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh.clone(), &av).map_err(napi_err)?;
  let bt = e.tensor_from_f32(sh, &bv).map_err(napi_err)?;
  Ok(cpu::mse_loss(&at, &bt).map_err(napi_err)? as f64)
}

#[napi(object)]
pub struct DropoutResult {
  pub output: Vec<f64>,
  pub mask: Vec<f64>,
}

#[napi]
pub fn dropout_op(a: Vec<f64>, shape: Vec<u32>, p: f64, seed: u32) -> napi::Result<DropoutResult> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  let (out, mask) = cpu::dropout(&at, p as f32, seed, dtype()).map_err(napi_err)?;
  Ok(DropoutResult {
    output: out.to_f32().iter().map(|&x| x as f64).collect(),
    mask: mask.iter().map(|&x| x as f64).collect(),
  })
}

#[napi]
pub fn embedding_op(
  weights: Vec<f64>,
  vocab_size: u32,
  embed_dim: u32,
  ids: Vec<i32>,
) -> napi::Result<Vec<f64>> {
  let wv: Vec<f32> = weights.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let wt = e
    .tensor_from_f32(vec![vocab_size as usize, embed_dim as usize], &wv)
    .map_err(napi_err)?;
  let out = cpu::embedding(&wt, &ids, dtype()).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn embedding_backward_op(
  grad_output: Vec<f64>,
  ids: Vec<i32>,
  vocab_size: u32,
  embed_dim: u32,
) -> Vec<f64> {
  let go: Vec<f32> = grad_output.iter().map(|&x| x as f32).collect();
  let gw = layers::embedding_backward(&go, &ids, vocab_size as usize, embed_dim as usize);
  gw.iter().map(|&x| x as f64).collect()
}

#[napi]
pub fn sum_all_op(a: Vec<f64>, shape: Vec<u32>) -> napi::Result<f64> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  Ok(cpu::sum_all(&at) as f64)
}

#[napi]
pub fn randn_op(shape: Vec<u32>, seed: u32) -> napi::Result<Vec<f64>> {
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let out = cpu::randn(sh, seed, neuralcabin_core::dtypes::DType::F32).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}

#[napi]
pub fn has_nan_or_inf_op(a: Vec<f64>, shape: Vec<u32>) -> napi::Result<bool> {
  let av: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  let sh: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
  let e = global_engine().read();
  let at = e.tensor_from_f32(sh, &av).map_err(napi_err)?;
  Ok(cpu::has_nan_or_inf(&at))
}

// ── Optimizer steps ───────────────────────────────────────────────────────────

#[napi(object)]
pub struct SgdResult {
  pub params: Vec<f64>,
  pub buf: Vec<f64>,
}

#[napi]
pub fn sgd_step(
  params: Vec<f64>,
  grads: Vec<f64>,
  buf: Vec<f64>,
  lr: f64,
  momentum: f64,
  weight_decay: f64,
) -> SgdResult {
  let pv: Vec<f32> = params.iter().map(|&x| x as f32).collect();
  let gv: Vec<f32> = grads.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = buf.iter().map(|&x| x as f32).collect();
  let (np, nb) = optim::sgd_step(&pv, &gv, &bv, lr as f32, momentum as f32, weight_decay as f32);
  SgdResult {
    params: np.iter().map(|&x| x as f64).collect(),
    buf: nb.iter().map(|&x| x as f64).collect(),
  }
}

#[napi(object)]
pub struct AdamResult {
  pub params: Vec<f64>,
  pub m: Vec<f64>,
  pub v: Vec<f64>,
}

#[napi]
pub fn adam_step(
  params: Vec<f64>,
  grads: Vec<f64>,
  m: Vec<f64>,
  v: Vec<f64>,
  lr: f64,
  beta1: f64,
  beta2: f64,
  eps: f64,
  t: u32,
) -> AdamResult {
  let pv: Vec<f32> = params.iter().map(|&x| x as f32).collect();
  let gv: Vec<f32> = grads.iter().map(|&x| x as f32).collect();
  let mv: Vec<f32> = m.iter().map(|&x| x as f32).collect();
  let vv: Vec<f32> = v.iter().map(|&x| x as f32).collect();
  let (np, nm, nv) = optim::adam_step(&pv, &gv, &mv, &vv, lr as f32, beta1 as f32, beta2 as f32, eps as f32, t);
  AdamResult {
    params: np.iter().map(|&x| x as f64).collect(),
    m: nm.iter().map(|&x| x as f64).collect(),
    v: nv.iter().map(|&x| x as f64).collect(),
  }
}

#[napi]
pub fn adamw_step(
  params: Vec<f64>,
  grads: Vec<f64>,
  m: Vec<f64>,
  v: Vec<f64>,
  lr: f64,
  beta1: f64,
  beta2: f64,
  eps: f64,
  weight_decay: f64,
  t: u32,
) -> AdamResult {
  let pv: Vec<f32> = params.iter().map(|&x| x as f32).collect();
  let gv: Vec<f32> = grads.iter().map(|&x| x as f32).collect();
  let mv: Vec<f32> = m.iter().map(|&x| x as f32).collect();
  let vv: Vec<f32> = v.iter().map(|&x| x as f32).collect();
  let (np, nm, nv) = optim::adamw_step(
    &pv, &gv, &mv, &vv, lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32, t,
  );
  AdamResult {
    params: np.iter().map(|&x| x as f64).collect(),
    m: nm.iter().map(|&x| x as f64).collect(),
    v: nv.iter().map(|&x| x as f64).collect(),
  }
}

#[napi]
pub fn clip_grad_norm(grads: Vec<f64>, max_norm: f64) -> Vec<f64> {
  let gv: Vec<f32> = grads.iter().map(|&x| x as f32).collect();
  optim::clip_grad_norm(&gv, max_norm as f32).iter().map(|&x| x as f64).collect()
}

// ── Q-Learning / DQN ─────────────────────────────────────────────────────────

#[napi]
pub fn epsilon_greedy(q_values: Vec<f64>, epsilon: f64, seed: u32) -> u32 {
  let qv: Vec<f32> = q_values.iter().map(|&x| x as f32).collect();
  rl::epsilon_greedy(&qv, epsilon as f32, seed) as u32
}

#[napi]
pub fn compute_td_targets(
  rewards: Vec<f64>,
  next_q_values: Vec<f64>,
  dones: Vec<f64>,
  gamma: f64,
  n_actions: u32,
) -> Vec<f64> {
  let rv: Vec<f32> = rewards.iter().map(|&x| x as f32).collect();
  let nqv: Vec<f32> = next_q_values.iter().map(|&x| x as f32).collect();
  let dv: Vec<f32> = dones.iter().map(|&x| x as f32).collect();
  rl::compute_td_targets(&rv, &nqv, &dv, gamma as f32, n_actions as usize)
    .iter()
    .map(|&x| x as f64)
    .collect()
}

#[napi(object)]
pub struct DqnLossResult {
  pub loss: f64,
  pub grad: Vec<f64>,
}

#[napi]
pub fn dqn_loss(
  q_values: Vec<f64>,
  actions: Vec<i32>,
  td_targets: Vec<f64>,
  n_actions: u32,
) -> DqnLossResult {
  let qv: Vec<f32> = q_values.iter().map(|&x| x as f32).collect();
  let tv: Vec<f32> = td_targets.iter().map(|&x| x as f32).collect();
  let (loss, grad) = rl::dqn_loss(&qv, &actions, &tv, n_actions as usize);
  DqnLossResult {
    loss: loss as f64,
    grad: grad.iter().map(|&x| x as f64).collect(),
  }
}

#[napi]
pub fn dqn_huber_loss(
  q_values: Vec<f64>,
  actions: Vec<i32>,
  td_targets: Vec<f64>,
  n_actions: u32,
  delta: f64,
) -> DqnLossResult {
  let qv: Vec<f32> = q_values.iter().map(|&x| x as f32).collect();
  let tv: Vec<f32> = td_targets.iter().map(|&x| x as f32).collect();
  let (loss, grad) = rl::dqn_huber_loss(&qv, &actions, &tv, n_actions as usize, delta as f32);
  DqnLossResult {
    loss: loss as f64,
    grad: grad.iter().map(|&x| x as f64).collect(),
  }
}

#[napi]
pub fn soft_update_target(mut target: Vec<f64>, online: Vec<f64>, tau: f64) -> Vec<f64> {
  let mut tv: Vec<f32> = target.iter().map(|&x| x as f32).collect();
  let ov: Vec<f32> = online.iter().map(|&x| x as f32).collect();
  rl::soft_update_target(&mut tv, &ov, tau as f32);
  tv.iter().map(|&x| x as f64).collect()
}

// ── Replay Buffer (stateless flat-array API for N-API compat) ────────────────

/// Sample a random mini-batch from flat replay buffer arrays.
/// Returns flat arrays for (states, actions, rewards, next_states, dones).
#[napi(object)]
pub struct ReplaySample {
  pub states: Vec<f64>,
  pub actions: Vec<f64>,
  pub rewards: Vec<f64>,
  pub next_states: Vec<f64>,
  pub dones: Vec<f64>,
}

#[napi]
pub fn replay_buffer_sample(
  states: Vec<f64>,
  actions: Vec<f64>,
  rewards: Vec<f64>,
  next_states: Vec<f64>,
  dones: Vec<f64>,
  size: u32,
  state_dim: u32,
  action_dim: u32,
  batch_size: u32,
  seed: u32,
) -> ReplaySample {
  let capacity = states.len() / state_dim as usize;
  let mut buf = rl::ReplayBuffer::new(capacity, state_dim as usize, action_dim as usize);
  buf.size = size as usize;

  // Restore existing data directly without push overhead.
  buf.states = states.iter().map(|&x| x as f32).collect();
  buf.actions = actions.iter().map(|&x| x as f32).collect();
  buf.rewards = rewards.iter().map(|&x| x as f32).collect();
  buf.next_states = next_states.iter().map(|&x| x as f32).collect();
  buf.dones = dones.iter().map(|&x| x as f32).collect();

  let (s, a, r, ns, d) = buf.sample(batch_size as usize, seed);
  ReplaySample {
    states: s.iter().map(|&x| x as f64).collect(),
    actions: a.iter().map(|&x| x as f64).collect(),
    rewards: r.iter().map(|&x| x as f64).collect(),
    next_states: ns.iter().map(|&x| x as f64).collect(),
    dones: d.iter().map(|&x| x as f64).collect(),
  }
}

// ── Neuroevolution ────────────────────────────────────────────────────────────

#[napi]
pub fn ne_mutate(params: Vec<f64>, p_mutate: f64, std: f64, seed: u32) -> Vec<f64> {
  let pv: Vec<f32> = params.iter().map(|&x| x as f32).collect();
  neuroevolution::mutate(&pv, p_mutate as f32, std as f32, seed)
    .iter()
    .map(|&x| x as f64)
    .collect()
}

#[napi]
pub fn ne_crossover_uniform(p1: Vec<f64>, p2: Vec<f64>, seed: u32) -> Vec<f64> {
  let pv1: Vec<f32> = p1.iter().map(|&x| x as f32).collect();
  let pv2: Vec<f32> = p2.iter().map(|&x| x as f32).collect();
  neuroevolution::crossover_uniform(&pv1, &pv2, seed)
    .iter()
    .map(|&x| x as f64)
    .collect()
}

#[napi]
pub fn ne_crossover_arithmetic(p1: Vec<f64>, p2: Vec<f64>, alpha: f64) -> Vec<f64> {
  let pv1: Vec<f32> = p1.iter().map(|&x| x as f32).collect();
  let pv2: Vec<f32> = p2.iter().map(|&x| x as f32).collect();
  neuroevolution::crossover_arithmetic(&pv1, &pv2, alpha as f32)
    .iter()
    .map(|&x| x as f64)
    .collect()
}

#[napi]
pub fn ne_tournament_select(fitnesses: Vec<f64>, k: u32, seed: u32) -> u32 {
  let fv: Vec<f32> = fitnesses.iter().map(|&x| x as f32).collect();
  neuroevolution::tournament_select(&fv, k as usize, seed) as u32
}

#[napi]
pub fn ne_truncation_select(fitnesses: Vec<f64>, n: u32) -> Vec<u32> {
  let fv: Vec<f32> = fitnesses.iter().map(|&x| x as f32).collect();
  neuroevolution::truncation_select(&fv, n as usize)
    .iter()
    .map(|&x| x as u32)
    .collect()
}

#[napi]
pub fn ne_evolve_generation(
  flat_params: Vec<f64>,
  fitnesses: Vec<f64>,
  param_count: u32,
  elite_count: u32,
  p_mutate: f64,
  mutation_std: f64,
  tournament_k: u32,
  seed: u32,
) -> Vec<f64> {
  let pv: Vec<f32> = flat_params.iter().map(|&x| x as f32).collect();
  let fv: Vec<f32> = fitnesses.iter().map(|&x| x as f32).collect();
  neuroevolution::evolve_generation(
    &pv,
    &fv,
    param_count as usize,
    elite_count as usize,
    p_mutate as f32,
    mutation_std as f32,
    tournament_k as usize,
    seed,
  )
  .iter()
  .map(|&x| x as f64)
  .collect()
}

#[napi(object)]
pub struct FitnessStats {
  pub min: f64,
  pub max: f64,
  pub mean: f64,
  pub std: f64,
}

#[napi]
pub fn ne_fitness_stats(fitnesses: Vec<f64>) -> FitnessStats {
  let fv: Vec<f32> = fitnesses.iter().map(|&x| x as f32).collect();
  let (min, max, mean, std) = neuroevolution::fitness_stats(&fv);
  FitnessStats { min: min as f64, max: max as f64, mean: mean as f64, std: std as f64 }
}

// ── Layer ops ─────────────────────────────────────────────────────────────────

#[napi]
pub fn linear_forward(
  input: Vec<f64>,
  b: u32,
  in_f: u32,
  weight: Vec<f64>,
  out_f: u32,
  bias: Vec<f64>,
) -> napi::Result<Vec<f64>> {
  let iv: Vec<f32> = input.iter().map(|&x| x as f32).collect();
  let wv: Vec<f32> = weight.iter().map(|&x| x as f32).collect();
  let bv: Vec<f32> = bias.iter().map(|&x| x as f32).collect();
  let e = global_engine().read();
  let dt = dtype();
  let it = e.tensor_from_f32(vec![b as usize, in_f as usize], &iv).map_err(napi_err)?;
  let wt = e.tensor_from_f32(vec![out_f as usize, in_f as usize], &wv).map_err(napi_err)?;
  let bt_opt = if bv.is_empty() {
    None
  } else {
    Some(e.tensor_from_f32(vec![out_f as usize], &bv).map_err(napi_err)?)
  };
  let out = layers::linear_forward(&it, &wt, bt_opt.as_ref(), dt).map_err(napi_err)?;
  Ok(out.to_f32().iter().map(|&x| x as f64).collect())
}
