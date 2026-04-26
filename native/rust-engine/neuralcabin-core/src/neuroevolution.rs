/// Neuroevolution — Selective Reproduction.
///
/// All weight vectors are flat Vec<f32>. The JS side owns model instances and
/// their full autograd-capable tensor state; Rust owns the evolutionary math.
use rayon::prelude::*;

// ── PRNG (Mulberry32) ─────────────────────────────────────────────────────────
#[inline]
fn mulberry32(s: &mut u32) -> f32 {
  *s = s.wrapping_add(0x6D2B79F5);
  let mut t = *s;
  t = (t ^ (t >> 15)).wrapping_mul(t | 1);
  t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
  ((t ^ (t >> 14)) as u64 & 0xFFFF_FFFF) as f32 / 4_294_967_296.0
}

fn normal_sample(rng: &mut u32) -> f32 {
  let u = loop { let v = mulberry32(rng); if v > 0.0 { break v; } };
  let v = loop { let v = mulberry32(rng); if v > 0.0 { break v; } };
  (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
}

// ── Mutation ──────────────────────────────────────────────────────────────────

/// Gaussian mutation: add N(0, std) perturbations with probability p_mutate.
pub fn mutate(params: &[f32], p_mutate: f32, std: f32, seed: u32) -> Vec<f32> {
  let mut rng = seed;
  params
    .iter()
    .map(|&p| {
      if mulberry32(&mut rng) < p_mutate {
        p + normal_sample(&mut rng) * std
      } else {
        p
      }
    })
    .collect()
}

/// Uniform-magnitude mutation: random sign flip + scale for all weights.
pub fn mutate_uniform(params: &[f32], p_mutate: f32, scale: f32, seed: u32) -> Vec<f32> {
  let mut rng = seed;
  params
    .iter()
    .map(|&p| {
      if mulberry32(&mut rng) < p_mutate {
        p + (mulberry32(&mut rng) * 2.0 - 1.0) * scale
      } else {
        p
      }
    })
    .collect()
}

// ── Crossover ─────────────────────────────────────────────────────────────────

/// Uniform crossover: each gene drawn independently from either parent.
pub fn crossover_uniform(p1: &[f32], p2: &[f32], seed: u32) -> Vec<f32> {
  assert_eq!(p1.len(), p2.len(), "crossover: parent length mismatch");
  let mut rng = seed;
  p1.iter().zip(p2.iter()).map(|(&a, &b)| if mulberry32(&mut rng) < 0.5 { a } else { b }).collect()
}

/// Single-point crossover: genes before the crossover point from parent 1,
/// remainder from parent 2.
pub fn crossover_single_point(p1: &[f32], p2: &[f32], seed: u32) -> Vec<f32> {
  assert_eq!(p1.len(), p2.len(), "crossover: parent length mismatch");
  let mut rng = seed;
  let cut = (mulberry32(&mut rng) * p1.len() as f32) as usize;
  let mut child = p1[..cut].to_vec();
  child.extend_from_slice(&p2[cut..]);
  child
}

/// Arithmetic crossover: gene = alpha*p1 + (1-alpha)*p2, alpha in [0,1].
pub fn crossover_arithmetic(p1: &[f32], p2: &[f32], alpha: f32) -> Vec<f32> {
  assert_eq!(p1.len(), p2.len(), "crossover: parent length mismatch");
  p1.par_iter().zip(p2.par_iter()).map(|(&a, &b)| alpha * a + (1.0 - alpha) * b).collect()
}

// ── Selection ─────────────────────────────────────────────────────────────────

/// Tournament selection: pick the fittest among `k` randomly drawn candidates.
/// Returns the index of the winner.
pub fn tournament_select(fitnesses: &[f32], k: usize, seed: u32) -> usize {
  assert!(!fitnesses.is_empty(), "tournament_select: empty fitness array");
  let mut rng = seed;
  let n = fitnesses.len();
  let mut best_idx = (mulberry32(&mut rng) * n as f32) as usize % n;
  for _ in 1..k {
    let idx = (mulberry32(&mut rng) * n as f32) as usize % n;
    if fitnesses[idx] > fitnesses[best_idx] {
      best_idx = idx;
    }
  }
  best_idx
}

/// Roulette-wheel (fitness-proportionate) selection.
/// Fitnesses must be non-negative. Returns selected index.
pub fn roulette_select(fitnesses: &[f32], seed: u32) -> usize {
  let total: f32 = fitnesses.iter().cloned().sum();
  if total == 0.0 {
    let mut rng = seed;
    return (mulberry32(&mut rng) * fitnesses.len() as f32) as usize % fitnesses.len();
  }
  let mut rng = seed;
  let mut r = mulberry32(&mut rng) * total;
  for (i, &f) in fitnesses.iter().enumerate() {
    r -= f;
    if r <= 0.0 {
      return i;
    }
  }
  fitnesses.len() - 1
}

/// Truncation selection: return indices of the top `n` individuals.
pub fn truncation_select(fitnesses: &[f32], n: usize) -> Vec<usize> {
  let mut idx: Vec<usize> = (0..fitnesses.len()).collect();
  idx.sort_unstable_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap_or(std::cmp::Ordering::Equal));
  idx.truncate(n);
  idx
}

// ── Full evolutionary generation step ────────────────────────────────────────

/// Run one generation of evolution.
///
/// `flat_params`: concatenated weight vectors of the whole population,
///   length = population_size * param_count.
/// `fitnesses`: fitness for each individual, length = population_size.
/// `elite_count`: how many top individuals survive unchanged.
/// Returns the new flat_params for the next generation.
pub fn evolve_generation(
  flat_params: &[f32],
  fitnesses: &[f32],
  param_count: usize,
  elite_count: usize,
  p_mutate: f32,
  mutation_std: f32,
  tournament_k: usize,
  seed: u32,
) -> Vec<f32> {
  let pop_size = fitnesses.len();
  assert_eq!(flat_params.len(), pop_size * param_count);

  // Sort individuals by fitness (descending) to determine elites.
  let elite_indices = truncation_select(fitnesses, elite_count.min(pop_size));

  let mut next = vec![0.0f32; pop_size * param_count];
  let mut rng = seed;

  for i in 0..pop_size {
    let child_slice = &mut next[i * param_count..(i + 1) * param_count];

    if i < elite_indices.len() {
      // Elite pass-through: copy the i-th fittest individual unchanged.
      let src = &flat_params[elite_indices[i] * param_count..(elite_indices[i] + 1) * param_count];
      child_slice.copy_from_slice(src);
      continue;
    }

    // Select two parents by tournament.
    let s1 = seed.wrapping_add(i as u32 * 1000007);
    let s2 = seed.wrapping_add(i as u32 * 2000003 + 1);
    let p1_idx = tournament_select(fitnesses, tournament_k, s1);
    let mut p2_idx = tournament_select(fitnesses, tournament_k, s2);
    if p2_idx == p1_idx { p2_idx = (p2_idx + 1) % pop_size; }

    let p1 = &flat_params[p1_idx * param_count..(p1_idx + 1) * param_count];
    let p2 = &flat_params[p2_idx * param_count..(p2_idx + 1) * param_count];

    // Uniform crossover → mutation.
    let cross_seed = seed.wrapping_add(i as u32 * 3000019);
    let mut child = crossover_uniform(p1, p2, cross_seed);
    let mut_seed = seed.wrapping_add(i as u32 * 4000037 + mulberry32(&mut rng).to_bits());
    child = mutate(&child, p_mutate, mutation_std, mut_seed);
    child_slice.copy_from_slice(&child);
  }

  next
}

// ── Fitness statistics ────────────────────────────────────────────────────────

/// Returns (min, max, mean, std) of the fitness array.
pub fn fitness_stats(fitnesses: &[f32]) -> (f32, f32, f32, f32) {
  if fitnesses.is_empty() {
    return (0.0, 0.0, 0.0, 0.0);
  }
  let n = fitnesses.len() as f32;
  let min = fitnesses.iter().cloned().fold(f32::INFINITY, f32::min);
  let max = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let mean = fitnesses.iter().sum::<f32>() / n;
  let var = fitnesses.iter().map(|&f| (f - mean) * (f - mean)).sum::<f32>() / n;
  (min, max, mean, var.sqrt())
}
