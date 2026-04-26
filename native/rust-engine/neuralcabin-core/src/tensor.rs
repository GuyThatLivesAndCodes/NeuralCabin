use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::dtypes::{DType, TensorStorage};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMeta {
  pub shape: Vec<usize>,
  pub dtype: DType,
}

#[derive(Clone, Debug)]
pub struct Tensor {
  pub meta: TensorMeta,
  pub storage: TensorStorage,
}

impl Tensor {
  pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
    let len = shape.iter().product();
    Self {
      meta: TensorMeta { shape, dtype },
      storage: TensorStorage::zeros(dtype, len),
    }
  }

  pub fn from_f32(shape: Vec<usize>, dtype: DType, src: &[f32]) -> Result<Self> {
    let len: usize = shape.iter().product();
    if len != src.len() {
      bail!("tensor element count mismatch: shape has {len}, src has {}", src.len());
    }
    Ok(Self {
      meta: TensorMeta { shape, dtype },
      storage: TensorStorage::from_f32(dtype, src),
    })
  }

  pub fn len(&self) -> usize {
    self.storage.len()
  }

  pub fn rank(&self) -> usize {
    self.meta.shape.len()
  }

  pub fn shape(&self) -> &[usize] {
    &self.meta.shape
  }

  pub fn dtype(&self) -> DType {
    self.meta.dtype
  }

  pub fn to_f32(&self) -> Vec<f32> {
    self.storage.to_f32()
  }

  pub fn cast(&self, dtype: DType) -> Self {
    if self.meta.dtype == dtype {
      return self.clone();
    }
    let f32buf = self.to_f32();
    Self {
      meta: TensorMeta {
        shape: self.meta.shape.clone(),
        dtype,
      },
      storage: TensorStorage::from_f32(dtype, &f32buf),
    }
  }

  pub fn update_from_f32(&mut self, src: &[f32]) -> Result<()> {
    if self.len() != src.len() {
      bail!("tensor update size mismatch: target {}, src {}", self.len(), src.len());
    }
    self.storage = TensorStorage::from_f32(self.meta.dtype, src);
    Ok(())
  }

  pub fn same_shape(&self, other: &Tensor) -> bool {
    self.meta.shape == other.meta.shape
  }
}
