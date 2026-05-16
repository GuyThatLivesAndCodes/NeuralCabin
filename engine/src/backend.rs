//! Burn backend selection for the engine.
//!
//! We define two backends so that:
//! - The CI / `cargo test` path runs on the CPU `NdArray` backend (no GPU needed).
//! - The production app picks `Wgpu` for laptop-class GPU acceleration.
//!
//! The training loop is generic over `AutodiffBackend`, so swapping backends
//! is a single type-alias change. See `src-tauri/src/lib.rs` for the WGPU
//! device instantiation.

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::tensor::backend::Backend;

/// CPU backend — used in unit tests so they run without a GPU driver.
pub type CpuBackend = NdArray<f32>;
/// CPU backend with reverse-mode autodiff wrapped around it (for tests).
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;

/// WGPU backend — used by the production training loop for GPU acceleration.
pub type GpuBackend = Wgpu<f32, i32>;
/// WGPU backend with autodiff (this is what training runs on).
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;

/// Re-export the GPU device type so the Tauri layer doesn't need to depend
/// on the Burn crate directly.
pub type GpuDevice = <GpuBackend as Backend>::Device;

/// Construct the default GPU device for training. The Tauri training loop
/// calls this once at the start of each run and reuses the device across all
/// batches.
pub fn default_gpu_device() -> GpuDevice {
    GpuDevice::default()
}

/// The default training backend selected at compile time.
///
/// `cargo test` overrides this via `#[cfg(test)]` to use the CPU backend so
/// the test suite doesn't require a GPU. The `lib.rs` re-exports do the same.
#[cfg(not(test))]
pub type DefaultBackend = GpuAutodiffBackend;
#[cfg(test)]
pub type DefaultBackend = CpuAutodiffBackend;
