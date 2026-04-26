use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AotExportConfig {
  pub model_name: String,
  pub opset: i64,
  pub producer_name: String,
}

impl Default for AotExportConfig {
  fn default() -> Self {
    Self {
      model_name: "neuralcabin-model".to_string(),
      opset: 18,
      producer_name: "neuralcabin-core".to_string(),
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OnnxExportManifest {
  pub format: String,
  pub model_name: String,
  pub opset: i64,
  pub producer_name: String,
  pub notes: Vec<String>,
}

pub fn export_aot_onnx_stub(path: &Path, cfg: &AotExportConfig) -> Result<()> {
  let manifest = OnnxExportManifest {
    format: "onnx".to_string(),
    model_name: cfg.model_name.clone(),
    opset: cfg.opset,
    producer_name: cfg.producer_name.clone(),
    notes: vec![
      "AOT export contract is active.".to_string(),
      "Full graph serialization is staged behind the same API.".to_string(),
    ],
  };

  let onnx_bytes = b"ONNX_STUB\0";
  fs::write(path, onnx_bytes).with_context(|| format!("failed to write {:?}", path))?;

  let sidecar = path.with_extension("onnx.json");
  let json = serde_json::to_vec_pretty(&manifest)?;
  fs::write(&sidecar, json).with_context(|| format!("failed to write {:?}", sidecar))?;
  Ok(())
}
