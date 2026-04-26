use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DType {
  F32,
  F16,
  BF16,
}

impl DType {
  pub fn bytes(self) -> usize {
    match self {
      DType::F32 => core::mem::size_of::<f32>(),
      DType::F16 | DType::BF16 => core::mem::size_of::<u16>(),
    }
  }
}

#[derive(Clone, Debug)]
pub enum TensorStorage {
  F32(Vec<f32>),
  F16(Vec<f16>),
  BF16(Vec<bf16>),
}

impl TensorStorage {
  pub fn zeros(dtype: DType, len: usize) -> Self {
    match dtype {
      DType::F32 => TensorStorage::F32(vec![0.0; len]),
      DType::F16 => TensorStorage::F16(vec![f16::from_f32(0.0); len]),
      DType::BF16 => TensorStorage::BF16(vec![bf16::from_f32(0.0); len]),
    }
  }

  pub fn from_f32(dtype: DType, src: &[f32]) -> Self {
    match dtype {
      DType::F32 => TensorStorage::F32(src.to_vec()),
      DType::F16 => TensorStorage::F16(src.iter().map(|v| f16::from_f32(*v)).collect()),
      DType::BF16 => TensorStorage::BF16(src.iter().map(|v| bf16::from_f32(*v)).collect()),
    }
  }

  pub fn len(&self) -> usize {
    match self {
      TensorStorage::F32(v) => v.len(),
      TensorStorage::F16(v) => v.len(),
      TensorStorage::BF16(v) => v.len(),
    }
  }

  pub fn to_f32(&self) -> Vec<f32> {
    match self {
      TensorStorage::F32(v) => v.clone(),
      TensorStorage::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
      TensorStorage::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
    }
  }
}
