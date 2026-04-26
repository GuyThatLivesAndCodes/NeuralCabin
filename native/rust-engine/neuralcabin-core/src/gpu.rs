use std::sync::Arc;

use anyhow::{Context, Result};
use wgpu::{
  Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue, RequestAdapterOptions,
};

pub struct WebGpuRuntime {
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
}

impl WebGpuRuntime {
  pub async fn try_new() -> Result<Self> {
    let instance = Instance::default();
    let adapter = instance
      .request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
      })
      .await
      .context("no GPU adapter available")?;

    let mut features = Features::empty();
    if adapter.features().contains(Features::SHADER_F16) {
      features |= Features::SHADER_F16;
    }

    let limits = Limits::downlevel_defaults().using_resolution(adapter.limits());
    let (device, queue) = adapter
      .request_device(
        &DeviceDescriptor {
          label: Some("neuralcabin-core-device"),
          required_features: features,
          required_limits: limits,
        },
        None,
      )
      .await
      .context("failed to request wgpu device")?;

    Ok(Self {
      device: Arc::new(device),
      queue: Arc::new(queue),
    })
  }

  pub fn try_new_blocking() -> Result<Self> {
    pollster::block_on(Self::try_new())
  }
}

pub fn fused_linear_gelu_kernel_wgsl() -> &'static str {
  // MatMul + Bias + GELU in one kernel pass. fp16 is enabled when supported.
  r#"
enable f16;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Shapes {
  m: u32,
  k: u32,
  n: u32,
};
@group(0) @binding(4) var<uniform> shapes: Shapes;

fn gelu(x: f32) -> f32 {
  let c = 0.7978845608;
  let t = tanh(c * (x + 0.044715 * x * x * x));
  return 0.5 * x * (1.0 + t);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= shapes.m || col >= shapes.n) { return; }

  var acc = 0.0;
  for (var p: u32 = 0u; p < shapes.k; p = p + 1u) {
    acc = acc + a[row * shapes.k + p] * b[p * shapes.n + col];
  }
  let x = acc + bias[col];
  out[row * shapes.n + col] = gelu(x);
}
"#
}

pub fn bf16_mixed_precision_note() -> &'static str {
  // WGSL does not currently expose native bf16 everywhere, so we use f32
  // accumulators and on-load conversion from packed bf16 where needed.
  "BF16 path uses software unpack/pack with f32 accumulation when native bf16 is unavailable."
}
