use std::sync::Arc;

use anyhow::{bail, Result};
use parking_lot::Mutex;

pub trait AllReduceBackend: Send + Sync {
  fn rank(&self) -> usize;
  fn world_size(&self) -> usize;
  fn allreduce_sum_f32(&self, tensor: &mut [f32]) -> Result<()>;
}

#[derive(Clone, Default)]
pub struct InProcessAllReduce {
  pub rank_id: usize,
  pub world: usize,
  slots: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl InProcessAllReduce {
  pub fn new(rank: usize, world: usize, slots: Arc<Mutex<Vec<Vec<f32>>>>) -> Self {
    Self {
      rank_id: rank,
      world,
      slots,
    }
  }
}

impl AllReduceBackend for InProcessAllReduce {
  fn rank(&self) -> usize {
    self.rank_id
  }

  fn world_size(&self) -> usize {
    self.world
  }

  fn allreduce_sum_f32(&self, tensor: &mut [f32]) -> Result<()> {
    let mut guard = self.slots.lock();
    if guard.len() != self.world {
      guard.resize_with(self.world, Vec::new);
    }
    guard[self.rank_id] = tensor.to_vec();

    if guard.iter().any(|slot| slot.is_empty()) {
      return Ok(());
    }

    let width = guard[0].len();
    if guard.iter().any(|slot| slot.len() != width) {
      bail!("allreduce width mismatch across ranks");
    }

    let mut acc = vec![0.0f32; width];
    for slot in guard.iter() {
      for (dst, src) in acc.iter_mut().zip(slot.iter()) {
        *dst += *src;
      }
    }
    tensor.copy_from_slice(&acc);

    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct RingAllReduceConfig {
  pub rank: usize,
  pub world_size: usize,
  pub peers: Vec<String>,
}

pub struct RingAllReduce {
  cfg: RingAllReduceConfig,
}

impl RingAllReduce {
  pub fn new(cfg: RingAllReduceConfig) -> Self {
    Self { cfg }
  }
}

impl AllReduceBackend for RingAllReduce {
  fn rank(&self) -> usize {
    self.cfg.rank
  }

  fn world_size(&self) -> usize {
    self.cfg.world_size
  }

  fn allreduce_sum_f32(&self, _tensor: &mut [f32]) -> Result<()> {
    // Network transport hook:
    // 1. scatter-reduce around the ring
    // 2. all-gather reduced chunks
    // 3. overlap comm/compute per chunk
    // This interface is intentionally stable so the JS/Electron side can
    // treat local and distributed training with the same API.
    bail!("ring allreduce transport is not wired yet")
  }
}
