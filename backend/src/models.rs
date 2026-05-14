use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub layers: Vec<LayerDef>,
    pub seed: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerDef {
    #[serde(rename = "linear")]
    Linear { in_dim: usize, out_dim: usize },
    #[serde(rename = "activation")]
    Activation { activation: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNetworkRequest {
    pub name: String,
    pub kind: String,
    pub seed: u64,
    pub layers: Vec<LayerDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListNetworksResponse {
    pub networks: Vec<Network>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub samples: usize,
    pub features: usize,
    pub labels: usize,
    pub task: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDatasetRequest {
    pub name: String,
    pub kind: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDatasetsResponse {
    pub datasets: Vec<Dataset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub kind: String,
    pub lr: f32,
    #[serde(default)]
    pub beta1: Option<f32>,
    #[serde(default)]
    pub beta2: Option<f32>,
    #[serde(default)]
    pub eps: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub optimizer: OptimizerConfig,
    pub loss: String,
    pub validation_frac: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartTrainingRequest {
    pub network_id: String,
    pub dataset_id: String,
    pub config: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartTrainingResponse {
    pub training_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingState {
    pub running: bool,
    pub paused: bool,
    pub stopped: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub last_loss: f32,
    pub last_val_loss: Option<f32>,
    pub last_accuracy: Option<f32>,
    pub loss_history: Vec<f32>,
    pub val_loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub error: Option<String>,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatusResponse {
    pub training_id: String,
    pub status: String,
    pub epoch: usize,
    pub total_epochs: usize,
    pub last_loss: f32,
    pub last_val_loss: Option<f32>,
    pub last_accuracy: Option<f32>,
    pub loss_history: Vec<f32>,
    pub val_loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    #[serde(rename = "epoch_update")]
    EpochUpdate {
        epoch: usize,
        total_epochs: usize,
        last_loss: f32,
        last_val_loss: Option<f32>,
        last_accuracy: Option<f32>,
        loss_history: Vec<f32>,
        val_loss_history: Vec<f32>,
        accuracy_history: Vec<f32>,
        elapsed_secs: f32,
    },
    #[serde(rename = "training_finished")]
    TrainingFinished {
        status: String,
        final_loss: f32,
        total_epochs: usize,
        elapsed_secs: f32,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}
