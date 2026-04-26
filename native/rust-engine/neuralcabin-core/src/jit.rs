use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StaticType {
  Tensor,
  Scalar,
  Bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypedExpr {
  TensorRef(String),
  Scalar(f32),
  Add(Box<TypedExpr>, Box<TypedExpr>),
  Mul(Box<TypedExpr>, Box<TypedExpr>),
  MatMul(Box<TypedExpr>, Box<TypedExpr>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypedStmt {
  Let {
    name: String,
    ty: StaticType,
    expr: TypedExpr,
  },
  Return(TypedExpr),
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TypedProgram {
  pub body: Vec<TypedStmt>,
}

pub struct NeuralScriptJit;

impl NeuralScriptJit {
  pub fn compile_to_spirv(program: &TypedProgram) -> Result<Vec<u32>> {
    let wgsl = lower_to_wgsl(program);
    let module = naga::front::wgsl::parse_str(&wgsl).context("WGSL parse failed in JIT")?;

    let mut validator = naga::valid::Validator::new(
      naga::valid::ValidationFlags::all(),
      naga::valid::Capabilities::empty(),
    );
    let info = validator.validate(&module).context("WGSL validation failed in JIT")?;

    let options = naga::back::spv::Options::default();
    let pipeline_options = naga::back::spv::PipelineOptions {
      shader_stage: naga::ShaderStage::Compute,
      entry_point: "main".to_string(),
    };
    let bounds = naga::proc::BoundsCheckPolicies::default();
    let spv = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options), &bounds)
      .context("SPIR-V emission failed in JIT")?;
    Ok(spv)
  }
}

fn lower_to_wgsl(_program: &TypedProgram) -> String {
  // Current stage: scalar pointwise entrypoint. The typed frontend already
  // catches architecture and tensor/shape category errors before lowering.
  // This gets replaced by graph-level kernel emission once fusion metadata is
  // attached to the TypedProgram IR.
  r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  output[i] = input_a[i] + input_b[i];
}
"#
  .to_string()
}
