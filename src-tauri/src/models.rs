use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ─── Network ────────────────────────────────────────────────────────────────

/// User-facing network kinds. The string `kind` on `Network` is one of these.
pub mod kinds {
    pub const FEEDFORWARD: &str = "feedforward";
    pub const NEXT_TOKEN:  &str = "next_token";

    pub fn all() -> [&'static str; 2] { [FEEDFORWARD, NEXT_TOKEN] }
    pub fn is_text(kind: &str) -> bool { kind == NEXT_TOKEN }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub layers: Vec<LayerDef>,
    pub seed: u64,
    pub input_dim: usize,
    pub output_dim: usize,
    pub created_at: DateTime<Utc>,
    /// Number of trainable parameters in the model. 0 until layers resolve.
    pub parameter_count: usize,
    /// True once the network has been trained at least once successfully.
    pub trained: bool,
    /// Sequence-model only: how many context tokens feed each prediction.
    /// None for feedforward.
    pub context_size: Option<usize>,
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
    pub input_dim: usize,
    /// Sequence networks only: context window size in tokens.
    #[serde(default)]
    pub context_size: Option<usize>,
}

// ─── Corpus / Dataset ───────────────────────────────────────────────────────

/// A dataset is the actual training data attached to a network. Shape depends
/// on network kind:
///   - feedforward:    rows of (features, targets) — `n × (in_dim + out_dim)`
///   - next_token:     a text body (pretraining) and/or input/output pairs
///                     (fine-tuning), plus the vocabulary used to encode them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    pub network_id: String,
    pub kind: String, // mirrors network.kind
    pub updated_at: DateTime<Utc>,

    // Feedforward
    #[serde(default)]
    pub feedforward: Option<FeedforwardCorpus>,

    // Next-token
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub pairs: Option<Vec<FineTunePair>>,
    #[serde(default)]
    pub vocab_mode: Option<String>, // "char" | "word"
    #[serde(default)]
    pub vocab: Option<Vec<String>>,
    #[serde(default)]
    pub stage: Option<String>, // "pretrain" | "finetune"
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedforwardCorpus {
    /// Row-major flat features, length = rows * in_dim
    pub features: Vec<f32>,
    /// Row-major flat targets, length = rows * out_dim
    pub targets: Vec<f32>,
    pub rows: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTunePair {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetCorpusRequest {
    pub network_id: String,
    #[serde(default)]
    pub feedforward: Option<FeedforwardCorpus>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub pairs: Option<Vec<FineTunePair>>,
    #[serde(default)]
    pub vocab_mode: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub kind: String,
    pub stage: Option<String>,

    // feedforward
    pub rows: Option<usize>,
    pub in_dim: Option<usize>,
    pub out_dim: Option<usize>,

    // next-token
    pub text_chars: Option<usize>,
    pub text_tokens: Option<usize>,
    pub pair_count: Option<usize>,
    pub vocab_size: Option<usize>,
    pub vocab_mode: Option<String>,
    pub training_examples: Option<usize>,
}

// ─── Training ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub kind: String, // "adam" | "sgd"
    pub lr: f32,
    #[serde(default)] pub beta1: Option<f32>,
    #[serde(default)] pub beta2: Option<f32>,
    #[serde(default)] pub eps: Option<f32>,
    #[serde(default)] pub momentum: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub optimizer: OptimizerConfig,
    pub loss: String, // "mse" | "crossentropy"
    pub seed: u64,
    /// Next-token only: mask user tokens in fine-tuning so the model is only
    /// scored on assistant output.
    #[serde(default)]
    pub mask_user_tokens: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartTrainingRequest {
    pub network_id: String,
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
    pub stopped: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub last_loss: f32,
    pub loss_history: Vec<f32>,
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
    pub loss_history: Vec<f32>,
    pub elapsed_secs: f32,
}

// Tauri event payloads emitted from Rust to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingUpdate {
    pub training_id: String,
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f32,
    pub loss_history: Vec<f32>,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFinished {
    pub training_id: String,
    pub status: String,
    pub final_loss: f32,
    pub total_epochs: usize,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingError {
    pub training_id: String,
    pub message: String,
}

// ─── Inference ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    pub network_id: String,
    /// Feedforward only: row-major flat features, length must equal in_dim.
    #[serde(default)] pub features: Option<Vec<f32>>,
    /// Next-token only: prompt text.
    #[serde(default)] pub prompt: Option<String>,
    /// Next-token only: how many tokens to generate.
    #[serde(default)] pub max_new_tokens: Option<usize>,
    /// Next-token only: sampling temperature (0 = greedy).
    #[serde(default)] pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferResponse {
    /// Feedforward: predicted output vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<f32>>,
    /// Next-token: generated text continuation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated: Option<String>,
    /// Next-token: per-step token + probability for transparency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<GenerationStep>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStep {
    pub token: String,
    pub probability: f32,
}
