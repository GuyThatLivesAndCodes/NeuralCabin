use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ─── Network ────────────────────────────────────────────────────────────────

pub mod kinds {
    pub const FEEDFORWARD: &str = "feedforward";
    pub const NEXT_TOKEN:  &str = "next_token";
    /// Decoder-only llama-style transformer. Produces GGUF files that load
    /// directly in llama.cpp / LM Studio (architecture = "llama").
    pub const TRANSFORMER: &str = "transformer";

    pub fn all() -> [&'static str; 3] { [FEEDFORWARD, NEXT_TOKEN, TRANSFORMER] }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerHParams {
    pub n_ctx:     usize,
    pub n_embd:    usize,
    pub n_layers:  usize,
    pub n_heads:   usize,
    pub n_ff:      usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f32,
}
fn default_rope_theta() -> f32 { 10000.0 }
fn default_rms_eps()    -> f32 { 1e-5 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub seed: u64,
    pub created_at: DateTime<Utc>,
    pub trained: bool,

    // For feedforward: fully specified at creation.
    // For next_token: zero until a vocabulary is built (then both dims are derived).
    pub input_dim: usize,
    pub output_dim: usize,
    /// Materialized full layer chain. Empty for next-token networks until
    /// vocabulary build wraps `hidden_layers` with input/output projections.
    pub layers: Vec<LayerDef>,
    /// Total parameter count once the model is materialized. 0 otherwise.
    pub parameter_count: usize,

    // Next-token only.
    pub hidden_layers: Option<Vec<LayerDef>>,
    pub context_size: Option<usize>,

    // Transformer only. Architecture hparams used to build the LM at create
    // time. The vocabulary is grown to size from the attached corpus; if its
    // real size differs from `vocab_size` here, training will rebuild.
    #[serde(default)]
    pub transformer: Option<TransformerHParams>,
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
    /// Feedforward: the full chain. Next-token: only the hidden chain
    /// (between input projection and output projection).
    pub layers: Vec<LayerDef>,
    /// Feedforward only.
    #[serde(default)]
    pub input_dim: Option<usize>,
    /// Next-token only.
    #[serde(default)]
    pub context_size: Option<usize>,
    /// Transformer only.
    #[serde(default)]
    pub transformer: Option<TransformerHParams>,
}

// ─── Vocabulary ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyInfo {
    pub mode: String,           // "char" | "subword" | "word" | "advanced"
    pub tokens: Vec<String>,
    pub options: VocabularyOptions,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VocabularyOptions {
    #[serde(default = "default_subword_merges")]
    pub subword_merges: usize,
    #[serde(default = "default_word_top_n")]
    pub word_top_n: usize,
}
fn default_subword_merges() -> usize { 200 }
fn default_word_top_n() -> usize { 500 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildVocabularyRequest {
    pub network_id: String,
    pub mode: String, // "char" | "subword" | "word"
    #[serde(default)]
    pub options: VocabularyOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetAdvancedVocabularyRequest {
    pub network_id: String,
    pub tokens: Vec<String>,
}

// ─── Corpus ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    pub network_id: String,
    pub kind: String,
    pub updated_at: DateTime<Utc>,

    // Feedforward.
    #[serde(default)] pub feedforward: Option<FeedforwardCorpus>,

    // Next-token.
    #[serde(default)] pub text: Option<String>,
    #[serde(default)] pub pairs: Option<Vec<FineTunePair>>,
    #[serde(default)] pub stage: Option<String>, // "pretrain" | "finetune"
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedforwardCorpus {
    pub features: Vec<f32>,
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
    #[serde(default)] pub feedforward: Option<FeedforwardCorpus>,
    #[serde(default)] pub text: Option<String>,
    #[serde(default)] pub pairs: Option<Vec<FineTunePair>>,
    #[serde(default)] pub stage: Option<String>,
    #[serde(default)] pub vocab_mode: Option<String>, // "char" | "word"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub kind: String,
    pub stage: Option<String>,

    pub rows: Option<usize>,
    pub in_dim: Option<usize>,
    pub out_dim: Option<usize>,

    pub text_chars: Option<usize>,
    pub text_tokens: Option<usize>,
    pub pair_count: Option<usize>,
    pub vocab_size: Option<usize>,
    pub vocab_mode: Option<String>,
    pub training_examples: Option<usize>,

    /// True if a vocab has been built. Required for next-token training.
    pub vocab_ready: bool,
    /// True if the model has been materialized (next-token only).
    pub model_ready: bool,
}

// ─── Training ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub kind: String, // "adam" | "sgd" | "adamw" | "lamb"
    pub lr: f32,
    #[serde(default)] pub beta1: Option<f32>,
    #[serde(default)] pub beta2: Option<f32>,
    #[serde(default)] pub eps: Option<f32>,
    #[serde(default)] pub momentum: Option<f32>,
    #[serde(default)] pub weight_decay: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub optimizer: OptimizerConfig,
    pub loss: String, // "mse" | "crossentropy"
    pub seed: u64,
    #[serde(default)] pub mask_user_tokens: Option<bool>,
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
    pub cancelled: bool,
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
    pub status: String, // "running" | "completed" | "cancelled" | "error"
    pub epoch: usize,
    pub total_epochs: usize,
    pub last_loss: f32,
    pub loss_history: Vec<f32>,
    pub elapsed_secs: f32,
}

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
    pub status: String, // "completed" | "cancelled"
    pub final_loss: f32,
    pub total_epochs: usize,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingError {
    pub training_id: String,
    pub message: String,
}

// ─── Training history ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigSummary {
    pub optimizer: String,
    pub lr: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub network_id: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    /// "completed" | "cancelled" | "aborted" | "error"
    pub status: String,
    pub config_summary: TrainingConfigSummary,
    pub total_epochs: usize,
    pub epochs_run: usize,
    pub final_loss: f32,
    pub elapsed_secs: f32,
    /// Downsampled to ≤500 points so state.json stays compact.
    pub loss_history: Vec<f32>,
}

// ─── Inference (streaming) ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "user" | "assistant"
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    pub network_id: String,
    /// Feedforward only.
    #[serde(default)] pub features: Option<Vec<f32>>,
    /// Next-token only.
    #[serde(default)] pub prompt: Option<String>,
    #[serde(default)] pub max_new_tokens: Option<usize>,
    #[serde(default)] pub temperature: Option<f32>,
    /// Multi-turn chat history. When supplied (and the network is a fine-tune
    /// model), the backend encodes every prior turn with `<user>` / `<assistant>`
    /// markers so the model has actual conversational context. The final turn
    /// must be `role: "user"` — the backend appends `<assistant>` and begins
    /// generation from there.
    #[serde(default)] pub messages: Option<Vec<ChatMessage>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferResponse {
    /// Synchronous result for feed-forward. None for streaming inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<f32>>,
    /// Inference job id for next-token streaming. None for feedforward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceToken {
    pub inference_id: String,
    pub index: usize,
    pub token: String,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceFinished {
    pub inference_id: String,
    pub status: String, // "completed" | "cancelled"
    pub generated: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // payload type only — emitted via Tauri events, never constructed directly
pub struct InferenceError {
    pub inference_id: String,
    pub message: String,
}
