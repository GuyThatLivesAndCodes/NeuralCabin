use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use memmap2::Mmap;

use crate::dtypes::DType;

#[derive(Debug)]
pub struct MmapDataset {
  path: PathBuf,
  mmap: Mmap,
  pub rows: usize,
  pub cols: usize,
  pub dtype: DType,
  row_bytes: usize,
}

impl MmapDataset {
  pub fn open<P: AsRef<Path>>(path: P, rows: usize, cols: usize, dtype: DType) -> Result<Self> {
    let path_buf = path.as_ref().to_path_buf();
    let file = File::open(&path_buf).with_context(|| format!("failed to open dataset file {:?}", path_buf))?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("failed to mmap {:?}", path_buf))?;

    let row_bytes = cols
      .checked_mul(dtype.bytes())
      .context("row byte count overflow")?;
    let expected = rows
      .checked_mul(row_bytes)
      .context("dataset byte size overflow")?;

    if mmap.len() < expected {
      bail!(
        "dataset is smaller than expected: got {} bytes, expected at least {expected} bytes",
        mmap.len()
      );
    }

    Ok(Self {
      path: path_buf,
      mmap,
      rows,
      cols,
      dtype,
      row_bytes,
    })
  }

  pub fn path(&self) -> &Path {
    &self.path
  }

  pub fn row_bytes(&self) -> usize {
    self.row_bytes
  }

  pub fn row_raw(&self, row: usize) -> Result<&[u8]> {
    if row >= self.rows {
      bail!("row out of range: {row} >= {}", self.rows);
    }
    let start = row * self.row_bytes;
    let end = start + self.row_bytes;
    Ok(&self.mmap[start..end])
  }

  pub fn row_f32(&self, row: usize) -> Result<Vec<f32>> {
    let bytes = self.row_raw(row)?;
    match self.dtype {
      DType::F32 => Ok(cast_slice::<u8, f32>(bytes).to_vec()),
      DType::F16 => {
        let half_words = cast_slice::<u8, u16>(bytes);
        Ok(half_words
          .iter()
          .map(|v| half::f16::from_bits(*v).to_f32())
          .collect())
      }
      DType::BF16 => {
        let half_words = cast_slice::<u8, u16>(bytes);
        Ok(half_words
          .iter()
          .map(|v| half::bf16::from_bits(*v).to_f32())
          .collect())
      }
    }
  }
}
