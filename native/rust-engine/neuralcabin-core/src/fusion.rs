use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpKind {
  MatMul,
  Add,
  Gelu,
  Relu,
  Softmax,
  LayerNorm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpNode {
  pub id: usize,
  pub kind: OpKind,
  pub inputs: Vec<usize>,
  pub output: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusionGroup {
  pub id: usize,
  pub ops: Vec<usize>,
  pub kernel_name: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FusionPlan {
  pub groups: Vec<FusionGroup>,
}

#[derive(Default)]
pub struct FusionPlanner;

impl FusionPlanner {
  pub fn plan(&self, nodes: &[OpNode]) -> FusionPlan {
    let mut groups = Vec::<FusionGroup>::new();
    let mut i = 0usize;

    while i < nodes.len() {
      if i + 2 < nodes.len()
        && nodes[i].kind == OpKind::MatMul
        && nodes[i + 1].kind == OpKind::Add
        && (nodes[i + 2].kind == OpKind::Gelu || nodes[i + 2].kind == OpKind::Relu)
      {
        let activation = if nodes[i + 2].kind == OpKind::Gelu { "gelu" } else { "relu" };
        groups.push(FusionGroup {
          id: groups.len(),
          ops: vec![nodes[i].id, nodes[i + 1].id, nodes[i + 2].id],
          kernel_name: format!("fused_linear_{activation}"),
        });
        i += 3;
        continue;
      }

      groups.push(FusionGroup {
        id: groups.len(),
        ops: vec![nodes[i].id],
        kernel_name: format!("op_{:?}", nodes[i].kind).to_lowercase(),
      });
      i += 1;
    }

    FusionPlan { groups }
  }
}
