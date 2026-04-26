/// Reinforcement Learning — Q-Learning / DQN support.
///
/// All state is passed in / out as flat Vec<f32> / primitive slices so that the
/// JS side owns lifetime and serialization. Rust owns only the hot arithmetic.
use rayon::prelude::*;

// ── Mulberry32 (local copy to keep rl.rs self-contained) ─────────────────────
#[inline]
fn mulberry32(s: &mut u32) -> f32 {
  *s = s.wrapping_add(0x6D2B79F5);
  let mut t = *s;
  t = (t ^ (t >> 15)).wrapping_mul(t | 1);
  t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
  ((t ^ (t >> 14)) as u64 & 0xFFFF_FFFF) as f32 / 4_294_967_296.0
}

// ── Replay Buffer ─────────────────────────────────────────────────────────────

/// Flat circular replay buffer. All transitions are stored as parallel flat
/// arrays so they can be passed directly across the N-API boundary without
/// extra (de)serialization.
pub struct ReplayBuffer {
  pub capacity: usize,
  pub state_dim: usize,
  pub action_dim: usize,
  pub size: usize,
  pub pos: usize,
  pub states: Vec<f32>,
  pub actions: Vec<f32>,
  pub rewards: Vec<f32>,
  pub next_states: Vec<f32>,
  pub dones: Vec<f32>,
}

impl ReplayBuffer {
  pub fn new(capacity: usize, state_dim: usize, action_dim: usize) -> Self {
    Self {
      capacity,
      state_dim,
      action_dim,
      size: 0,
      pos: 0,
      states: vec![0.0; capacity * state_dim],
      actions: vec![0.0; capacity * action_dim],
      rewards: vec![0.0; capacity],
      next_states: vec![0.0; capacity * state_dim],
      dones: vec![0.0; capacity],
    }
  }

  /// Push one (s, a, r, s', done) transition. Overwrites oldest when full.
  pub fn push(
    &mut self,
    state: &[f32],
    action: &[f32],
    reward: f32,
    next_state: &[f32],
    done: bool,
  ) {
    let p = self.pos;
    let sd = self.state_dim;
    let ad = self.action_dim;
    self.states[p * sd..(p + 1) * sd].copy_from_slice(state);
    self.actions[p * ad..(p + 1) * ad].copy_from_slice(action);
    self.rewards[p] = reward;
    self.next_states[p * sd..(p + 1) * sd].copy_from_slice(next_state);
    self.dones[p] = if done { 1.0 } else { 0.0 };
    self.pos = (self.pos + 1) % self.capacity;
    self.size = (self.size + 1).min(self.capacity);
  }

  /// Sample a random mini-batch. Returns (states, actions, rewards, next_states, dones)
  /// each as a flat Vec<f32>.
  pub fn sample(
    &self,
    batch_size: usize,
    seed: u32,
  ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = self.size;
    let sd = self.state_dim;
    let ad = self.action_dim;
    let mut rng = seed;

    let mut s_out = Vec::with_capacity(batch_size * sd);
    let mut a_out = Vec::with_capacity(batch_size * ad);
    let mut r_out = Vec::with_capacity(batch_size);
    let mut ns_out = Vec::with_capacity(batch_size * sd);
    let mut d_out = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
      let idx = (mulberry32(&mut rng) * n as f32) as usize % n;
      s_out.extend_from_slice(&self.states[idx * sd..(idx + 1) * sd]);
      a_out.extend_from_slice(&self.actions[idx * ad..(idx + 1) * ad]);
      r_out.push(self.rewards[idx]);
      ns_out.extend_from_slice(&self.next_states[idx * sd..(idx + 1) * sd]);
      d_out.push(self.dones[idx]);
    }

    (s_out, a_out, r_out, ns_out, d_out)
  }
}

// ── Epsilon-greedy action selection ──────────────────────────────────────────

/// Returns the selected action index.
/// q_values: flat [n_actions]; epsilon: exploration probability.
pub fn epsilon_greedy(q_values: &[f32], epsilon: f32, seed: u32) -> usize {
  let mut rng = seed;
  if mulberry32(&mut rng) < epsilon {
    // Random action
    (mulberry32(&mut rng) * q_values.len() as f32) as usize % q_values.len()
  } else {
    // Greedy
    q_values
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
      .map(|(i, _)| i)
      .unwrap_or(0)
  }
}

// ── TD target computation ─────────────────────────────────────────────────────

/// Compute DQN TD targets: r + gamma * max(Q(s')) * (1 - done).
/// next_q_values: flat [batch, n_actions].
/// Returns td_targets: [batch].
pub fn compute_td_targets(
  rewards: &[f32],
  next_q_values: &[f32],
  dones: &[f32],
  gamma: f32,
  n_actions: usize,
) -> Vec<f32> {
  let batch = rewards.len();
  (0..batch)
    .into_par_iter()
    .map(|i| {
      let row = &next_q_values[i * n_actions..(i + 1) * n_actions];
      let max_q = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
      rewards[i] + gamma * max_q * (1.0 - dones[i])
    })
    .collect()
}

/// DQN loss: MSE between current Q predictions and TD targets.
/// q_values: flat [batch, n_actions]; actions: [batch] integer indices;
/// td_targets: [batch].
/// Returns (loss_scalar, grad_q_values flat [batch, n_actions]).
pub fn dqn_loss(
  q_values: &[f32],
  actions: &[i32],
  td_targets: &[f32],
  n_actions: usize,
) -> (f32, Vec<f32>) {
  let batch = actions.len();
  let mut loss = 0.0f32;
  let mut grad = vec![0.0f32; batch * n_actions];

  for i in 0..batch {
    let a = actions[i] as usize;
    if a >= n_actions { continue; }
    let q_a = q_values[i * n_actions + a];
    let err = q_a - td_targets[i];
    loss += err * err;
    grad[i * n_actions + a] = 2.0 * err / batch as f32;
  }

  (loss / batch as f32, grad)
}

// ── Huber / clipped TD error ──────────────────────────────────────────────────

/// Huber loss variant for more stable DQN training.
/// Returns (loss, grad) as above.
pub fn dqn_huber_loss(
  q_values: &[f32],
  actions: &[i32],
  td_targets: &[f32],
  n_actions: usize,
  delta: f32,
) -> (f32, Vec<f32>) {
  let batch = actions.len();
  let mut loss = 0.0f32;
  let mut grad = vec![0.0f32; batch * n_actions];

  for i in 0..batch {
    let a = actions[i] as usize;
    if a >= n_actions { continue; }
    let q_a = q_values[i * n_actions + a];
    let err = q_a - td_targets[i];
    let abs_err = err.abs();
    if abs_err <= delta {
      loss += 0.5 * err * err;
      grad[i * n_actions + a] = err / batch as f32;
    } else {
      loss += delta * (abs_err - 0.5 * delta);
      grad[i * n_actions + a] = delta * err.signum() / batch as f32;
    }
  }

  (loss / batch as f32, grad)
}

// ── Target network soft update ─────────────────────────────────────────────────

/// Polyak averaging: target = (1-tau)*target + tau*online.
pub fn soft_update_target(target: &mut [f32], online: &[f32], tau: f32) {
  target.par_iter_mut().zip(online.par_iter()).for_each(|(t, &o)| {
    *t = (1.0 - tau) * *t + tau * o;
  });
}
