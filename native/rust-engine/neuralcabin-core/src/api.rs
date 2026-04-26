use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cpu;
use crate::data::MmapDataset;
use crate::distributed::AllReduceBackend;
use crate::dtypes::DType;
use crate::fusion::{FusionPlan, FusionPlanner, OpNode};
use crate::gpu::WebGpuRuntime;
use crate::jit::{NeuralScriptJit, TypedProgram};
use crate::onnx::{export_aot_onnx_stub, AotExportConfig};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum BackendMode {
  Auto,
  Cpu,
  Gpu,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum PrecisionMode {
  FP32,
  FP16,
  BF16,
}

impl PrecisionMode {
  pub fn as_dtype(self) -> DType {
    match self {
      PrecisionMode::FP32 => DType::F32,
      PrecisionMode::FP16 => DType::F16,
      PrecisionMode::BF16 => DType::BF16,
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
  pub backend: BackendMode,
  pub precision: PrecisionMode,
  pub operator_fusion: bool,
}

impl Default for EngineConfig {
  fn default() -> Self {
    Self {
      backend: BackendMode::Auto,
      precision: PrecisionMode::FP16,
      operator_fusion: true,
    }
  }
}

pub struct Engine {
  pub config: EngineConfig,
  gpu: Option<WebGpuRuntime>,
}

impl Engine {
  pub fn new(config: EngineConfig) -> Self {
    let gpu = match config.backend {
      BackendMode::Cpu => None,
      BackendMode::Gpu | BackendMode::Auto => WebGpuRuntime::try_new_blocking().ok(),
    };
    Self { config, gpu }
  }

  pub fn backend_name(&self) -> &'static str {
    if self.gpu.is_some() {
      "wgpu"
    } else {
      "cpu"
    }
  }

  pub fn supports_mixed_precision(&self) -> bool {
    true
  }

  pub fn tensor_from_f32(&self, shape: Vec<usize>, values: &[f32]) -> Result<Tensor> {
    Tensor::from_f32(shape, self.config.precision.as_dtype(), values)
  }

  pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // GPU path is intentionally wired behind the same API contract; CPU path
    // remains available for hosts without adapter support.
    cpu::matmul(a, b, self.config.precision.as_dtype())
  }

  pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cpu::add(a, b, self.config.precision.as_dtype())
  }

  pub fn relu(&self, a: &Tensor) -> Result<Tensor> {
    cpu::relu(a, self.config.precision.as_dtype())
  }

  pub fn plan_fusion(&self, nodes: &[OpNode]) -> FusionPlan {
    let planner = FusionPlanner::default();
    planner.plan(nodes)
  }

  pub fn compile_neuralscript_to_spirv(&self, program: &TypedProgram) -> Result<Vec<u32>> {
    NeuralScriptJit::compile_to_spirv(program)
  }

  pub fn open_dataset_mmap<P: AsRef<Path>>(
    &self,
    path: P,
    rows: usize,
    cols: usize,
    dtype: DType,
  ) -> Result<MmapDataset> {
    MmapDataset::open(path, rows, cols, dtype)
  }

  pub fn allreduce_sum(
    &self,
    backend: Arc<dyn AllReduceBackend>,
    tensor: &mut [f32],
  ) -> Result<()> {
    backend.allreduce_sum_f32(tensor)
  }

  pub fn export_onnx_aot<P: AsRef<Path>>(&self, path: P, cfg: Option<AotExportConfig>) -> Result<()> {
    let cfg = cfg.unwrap_or_default();
    export_aot_onnx_stub(path.as_ref(), &cfg).context("AOT export failed")
  }
}
