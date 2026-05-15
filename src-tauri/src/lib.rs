mod models;
mod persistence;

use models::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use chrono::Utc;
use neuralcabin_engine::nn::{LayerSpec, Model};
use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
use neuralcabin_engine::tensor::{SplitMix64, Tensor};
use neuralcabin_engine::tokenizer::{
    TokenizerMode, Vocabulary, VocabularyOptions as EngineVocabularyOptions,
    EOS_ID,
};
use neuralcabin_engine::corpus::{
    build_finetuning_tensors, build_pretraining_tensors, encode_context, Pair,
};
use neuralcabin_engine::activations::softmax_rows;
use neuralcabin_engine::{Activation, Loss};

// ─── State ──────────────────────────────────────────────────────────────────

pub struct AppState {
    pub(crate) networks:  Arc<RwLock<HashMap<String, Network>>>,
    pub(crate) models:    Arc<RwLock<HashMap<String, Arc<RwLock<Model>>>>>,
    pub(crate) vocabs:    Arc<RwLock<HashMap<String, VocabEntry>>>,
    pub(crate) corpora:   Arc<RwLock<HashMap<String, Corpus>>>,
    pub(crate) trainers:  Arc<RwLock<HashMap<String, TrainerHandle>>>,
    pub(crate) inferrers: Arc<RwLock<HashMap<String, InferenceHandle>>>,
    /// Where to write `state.json`. None disables persistence — used by unit
    /// tests that don't go through the Tauri setup hook.
    pub(crate) data_dir:  Arc<RwLock<Option<PathBuf>>>,
}

#[derive(Clone)]
pub(crate) struct VocabEntry {
    pub(crate) vocab: Arc<RwLock<Vocabulary>>,
    pub(crate) info: VocabularyInfo,
}

#[derive(Clone)]
struct TrainerHandle {
    state:  Arc<RwLock<TrainingState>>,
    cancel: Arc<AtomicBool>,
    /// Set together with `cancel` by `abort_training` — tells the training
    /// loop to restore the pre-training model snapshot on the way out.
    rollback: Arc<AtomicBool>,
}

#[derive(Clone)]
struct InferenceHandle {
    cancel: Arc<AtomicBool>,
}

impl AppState {
    pub(crate) fn new() -> Self {
        Self {
            networks:  Arc::new(RwLock::new(HashMap::new())),
            models:    Arc::new(RwLock::new(HashMap::new())),
            vocabs:    Arc::new(RwLock::new(HashMap::new())),
            corpora:   Arc::new(RwLock::new(HashMap::new())),
            trainers:  Arc::new(RwLock::new(HashMap::new())),
            inferrers: Arc::new(RwLock::new(HashMap::new())),
            data_dir:  Arc::new(RwLock::new(None)),
        }
    }
}

/// Save the current state to disk. Most command handlers call this after a
/// mutation. Errors are returned to the caller so the UI can surface them —
/// silent persistence failures are how data gets lost.
async fn persist(state: &AppState) -> Result<(), String> {
    let Some(dir) = state.data_dir.read().await.clone() else { return Ok(()); };
    // Timing diagnostics: if "save takes forever" is reported, this log line
    // pins down whether the cost is in snapshotting, JSON encoding, or the
    // disk write itself.
    let started = std::time::Instant::now();
    let result = persistence::save_to_dir(state, &dir).await;
    let elapsed = started.elapsed();
    if elapsed.as_millis() > 500 {
        eprintln!("[neuralcabin] persist took {}ms — investigating slow save",
                  elapsed.as_millis());
    }
    result
}

// ─── Parsing helpers ────────────────────────────────────────────────────────

fn parse_activation(name: &str) -> Result<Activation, String> {
    match name.to_ascii_lowercase().as_str() {
        "identity" => Ok(Activation::Identity),
        "relu"     => Ok(Activation::ReLU),
        "sigmoid"  => Ok(Activation::Sigmoid),
        "tanh"     => Ok(Activation::Tanh),
        "softmax"  => Ok(Activation::Softmax),
        other => Err(format!("unknown activation '{other}'")),
    }
}

fn parse_loss(name: &str) -> Result<Loss, String> {
    match name.to_ascii_lowercase().as_str() {
        "mse" | "mean_squared_error" | "meansquarederror" => Ok(Loss::MeanSquaredError),
        "ce"  | "crossentropy" | "cross_entropy"          => Ok(Loss::CrossEntropy),
        other => Err(format!("unknown loss '{other}'")),
    }
}

fn parse_tokenizer_mode(name: &str) -> Result<TokenizerMode, String> {
    match name.to_ascii_lowercase().as_str() {
        "char"     => Ok(TokenizerMode::Char),
        "subword"  => Ok(TokenizerMode::Subword),
        "word"     => Ok(TokenizerMode::Word),
        "advanced" => Ok(TokenizerMode::Advanced),
        other => Err(format!(
            "unknown vocab mode '{other}' (expected char|subword|word|advanced)"
        )),
    }
}

fn build_optimizer(cfg: &OptimizerConfig, shapes: &[Vec<usize>]) -> Result<Optimizer, String> {
    let kind = match cfg.kind.to_ascii_lowercase().as_str() {
        "adam" => OptimizerKind::Adam {
            lr: cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
        },
        "adamw" => OptimizerKind::AdamW {
            lr: cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
            weight_decay: cfg.weight_decay.unwrap_or(0.01),
        },
        "lamb" => OptimizerKind::Lamb {
            lr: cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
            weight_decay: cfg.weight_decay.unwrap_or(0.0),
        },
        "sgd" => OptimizerKind::Sgd {
            lr: cfg.lr,
            momentum: cfg.momentum.unwrap_or(0.0),
        },
        other => return Err(format!("unknown optimizer '{other}' (expected adam|adamw|lamb|sgd)")),
    };
    Ok(Optimizer::new(kind, shapes))
}

fn build_layer_specs(layers: &[LayerDef], input_dim: usize) -> Result<(Vec<LayerSpec>, usize), String> {
    let mut specs = Vec::with_capacity(layers.len());
    let mut cur = input_dim;
    for (i, l) in layers.iter().enumerate() {
        match l {
            LayerDef::Linear { in_dim, out_dim } => {
                if *in_dim != cur {
                    return Err(format!(
                        "layer {i}: linear in_dim {in_dim} doesn't match running dim {cur}"
                    ));
                }
                if *out_dim == 0 {
                    return Err(format!("layer {i}: linear out_dim must be > 0"));
                }
                specs.push(LayerSpec::Linear { in_dim: *in_dim, out_dim: *out_dim });
                cur = *out_dim;
            }
            LayerDef::Activation { activation } => {
                specs.push(LayerSpec::Activation(parse_activation(activation)?));
            }
        }
    }
    Ok((specs, cur))
}

// ─── Network commands ───────────────────────────────────────────────────────

#[tauri::command]
async fn create_network(
    state: State<'_, AppState>,
    req: CreateNetworkRequest,
) -> Result<Network, String> {
    if !kinds::all().contains(&req.kind.as_str()) {
        return Err(format!(
            "unknown network kind '{}' (expected one of: {:?})", req.kind, kinds::all()
        ));
    }
    if req.layers.is_empty() && req.kind == kinds::FEEDFORWARD {
        return Err("feedforward network must have at least one layer".into());
    }
    let name = req.name.trim().to_string();
    if name.is_empty() { return Err("network name cannot be empty".into()); }

    let id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let network = if req.kind == kinds::FEEDFORWARD {
        let input_dim = req.input_dim.unwrap_or(0);
        if input_dim == 0 {
            return Err("feedforward network requires input_dim > 0".into());
        }
        let (specs, output_dim) = build_layer_specs(&req.layers, input_dim)?;
        let model = Model::from_specs(input_dim, &specs, req.seed);
        let parameter_count = model.parameter_count();
        state.models.write().await.insert(id.clone(), Arc::new(RwLock::new(model)));

        Network {
            id: id.clone(),
            name, kind: req.kind, seed: req.seed, created_at: now, trained: false,
            input_dim, output_dim,
            layers: req.layers, parameter_count,
            hidden_layers: None, context_size: None,
        }
    } else {
        // Next-token: only the hidden chain is meaningful at this point.
        let context_size = req.context_size.unwrap_or(0);
        if context_size == 0 {
            return Err("next-token network requires context_size > 0".into());
        }
        Network {
            id: id.clone(),
            name, kind: req.kind, seed: req.seed, created_at: now, trained: false,
            input_dim: 0, output_dim: 0,
            layers: Vec::new(), parameter_count: 0,
            hidden_layers: Some(req.layers),
            context_size: Some(context_size),
        }
    };

    state.networks.write().await.insert(id, network.clone());
    persist(&state).await?;
    Ok(network)
}

#[tauri::command]
async fn list_networks(state: State<'_, AppState>) -> Result<Vec<Network>, String> {
    Ok(state.networks.read().await.values().cloned().collect())
}

#[tauri::command]
async fn get_network(state: State<'_, AppState>, id: String) -> Result<Network, String> {
    state.networks.read().await.get(&id).cloned().ok_or_else(|| "Network not found".into())
}

#[tauri::command]
async fn delete_network(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    state.models.write().await.remove(&id);
    state.vocabs.write().await.remove(&id);
    state.corpora.write().await.remove(&id);
    let removed = state.networks.write().await.remove(&id).is_some();
    persist(&state).await?;
    Ok(removed)
}

// ─── Corpus commands ────────────────────────────────────────────────────────

#[tauri::command]
async fn set_corpus(
    state: State<'_, AppState>,
    req: SetCorpusRequest,
) -> Result<CorpusStats, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;

    let mut corpus = Corpus {
        network_id: net.id.clone(),
        kind: net.kind.clone(),
        updated_at: Utc::now(),
        feedforward: None,
        text: None,
        pairs: None,
        stage: None,
    };

    if net.kind == kinds::FEEDFORWARD {
        let ff = req.feedforward.ok_or_else(|| "feedforward corpus required".to_string())?;
        validate_feedforward(&ff, &net)?;
        corpus.feedforward = Some(ff);
    } else if net.kind == kinds::NEXT_TOKEN {
        corpus.text  = req.text;
        corpus.pairs = req.pairs;
        corpus.stage = req.stage;
    } else {
        return Err(format!("unsupported network kind '{}'", net.kind));
    }

    // Persist the corpus first so subsequent operations see it.
    state.corpora.write().await.insert(net.id.clone(), corpus.clone());

    // For next-token networks, auto-build the vocabulary from the new corpus so
    // the user doesn't have to do it manually. Without this, training would
    // immediately fail with "no vocabulary" after saving a corpus.
    if net.kind == kinds::NEXT_TOKEN {
        let mode_name = req.vocab_mode.as_deref().unwrap_or("char");
        if let Ok(mode) = parse_tokenizer_mode(mode_name) {
            if !matches!(mode, TokenizerMode::Advanced) {
                let mut texts: Vec<String> = Vec::new();
                if let Some(t) = &corpus.text { texts.push(t.clone()); }
                if let Some(ps) = &corpus.pairs {
                    for p in ps { texts.push(p.input.clone()); texts.push(p.output.clone()); }
                }
                if !texts.is_empty() {
                    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                    let opts = EngineVocabularyOptions::default();
                    let vocab = Vocabulary::build(mode, &refs, &opts);
                    let vocab_size = vocab.size();
                    let info = VocabularyInfo {
                        mode: mode_str(mode),
                        tokens: vocab.tokens.clone(),
                        options: VocabularyOptions::default(),
                        updated_at: Utc::now(),
                    };
                    let entry = VocabEntry {
                        vocab: Arc::new(RwLock::new(vocab)),
                        info,
                    };
                    state.vocabs.write().await.insert(net.id.clone(), entry);
                    rebuild_next_token_model(&state, &net, vocab_size).await?;
                }
            }
        }
    }

    // Recompute stats AFTER any vocab/model rebuild so the response reflects them.
    let net_after = state.networks.read().await.get(&req.network_id).cloned().unwrap_or(net);
    let stats = compute_stats(
        &net_after,
        &corpus,
        &*state.vocabs.read().await,
        &*state.models.read().await,
    ).await;
    persist(&state).await?;
    Ok(stats)
}

fn validate_feedforward(ff: &FeedforwardCorpus, net: &Network) -> Result<(), String> {
    if ff.in_dim != net.input_dim {
        return Err(format!(
            "feedforward in_dim {} doesn't match network input_dim {}",
            ff.in_dim, net.input_dim
        ));
    }
    if ff.out_dim != net.output_dim {
        return Err(format!(
            "feedforward out_dim {} doesn't match network output_dim {}",
            ff.out_dim, net.output_dim
        ));
    }
    if ff.features.len() != ff.rows * ff.in_dim {
        return Err("feedforward features length doesn't match rows × in_dim".into());
    }
    if ff.targets.len() != ff.rows * ff.out_dim {
        return Err("feedforward targets length doesn't match rows × out_dim".into());
    }
    if ff.rows == 0 {
        return Err("feedforward corpus must have at least one row".into());
    }
    Ok(())
}

async fn compute_stats(
    net: &Network,
    corpus: &Corpus,
    vocabs: &HashMap<String, VocabEntry>,
    models: &HashMap<String, Arc<RwLock<Model>>>,
) -> CorpusStats {
    let mut stats = CorpusStats {
        kind: net.kind.clone(),
        stage: corpus.stage.clone(),
        rows: None, in_dim: None, out_dim: None,
        text_chars: None, text_tokens: None, pair_count: None,
        vocab_size: None, vocab_mode: None,
        training_examples: None,
        vocab_ready: false,
        model_ready: models.contains_key(&net.id),
    };

    if let Some(ff) = &corpus.feedforward {
        stats.rows    = Some(ff.rows);
        stats.in_dim  = Some(ff.in_dim);
        stats.out_dim = Some(ff.out_dim);
        stats.training_examples = Some(ff.rows);
    }

    if net.kind == kinds::NEXT_TOKEN {
        if let Some(text) = &corpus.text {
            stats.text_chars = Some(text.chars().count());
        }
        stats.pair_count = corpus.pairs.as_ref().map(|p| p.len());

        if let Some(entry) = vocabs.get(&net.id) {
            stats.vocab_ready = true;
            stats.vocab_mode = Some(entry.info.mode.clone());
            let v = entry.vocab.read().await;
            stats.vocab_size = Some(v.size());
            if let Some(text) = &corpus.text {
                stats.text_tokens = Some(v.encode(text).len());
            }
            let context_size = net.context_size.unwrap_or(0);
            if context_size > 0 {
                let mut total = 0usize;
                if let Some(text) = &corpus.text {
                    let n = v.encode(text).len();
                    if n > context_size { total += n - context_size; }
                }
                if let Some(pairs) = &corpus.pairs {
                    for p in pairs {
                        let len = 1 + v.encode(&p.input).len() + 1
                                + v.encode(&p.output).len() + 1;
                        if len >= 2 { total += len - 1; }
                    }
                }
                stats.training_examples = Some(total);
            }
        }
    }

    stats
}

#[tauri::command]
async fn get_corpus(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<Corpus>, String> {
    Ok(state.corpora.read().await.get(&network_id).cloned())
}

#[tauri::command]
async fn corpus_stats(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<CorpusStats>, String> {
    let net = state.networks.read().await.get(&network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let corpora = state.corpora.read().await;
    let Some(corpus) = corpora.get(&network_id) else { return Ok(None); };
    let vocabs = state.vocabs.read().await;
    let models = state.models.read().await;
    let stats = compute_stats(&net, corpus, &vocabs, &models).await;
    Ok(Some(stats))
}

// ─── Vocabulary commands ────────────────────────────────────────────────────

#[tauri::command]
async fn build_vocabulary(
    state: State<'_, AppState>,
    req: BuildVocabularyRequest,
) -> Result<VocabularyInfo, String> {
    let mode = parse_tokenizer_mode(&req.mode)?;
    if matches!(mode, TokenizerMode::Advanced) {
        return Err("use set_advanced_vocabulary for advanced mode".to_string());
    }
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    if net.kind != kinds::NEXT_TOKEN {
        return Err("vocabularies apply only to next-token networks".to_string());
    }
    let corpus = state.corpora.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "No corpus attached. Add training data on the Corpus tab first.".to_string())?;

    let mut owned: Vec<String> = Vec::new();
    if let Some(t) = &corpus.text { owned.push(t.clone()); }
    if let Some(ps) = &corpus.pairs {
        for p in ps { owned.push(p.input.clone()); owned.push(p.output.clone()); }
    }
    if owned.is_empty() {
        return Err("corpus is empty — add text or pairs before building a vocabulary".to_string());
    }
    let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();

    let engine_opts = EngineVocabularyOptions {
        subword_merges: req.options.subword_merges,
        word_top_n:     req.options.word_top_n,
    };
    let vocab = Vocabulary::build(mode, &refs, &engine_opts);
    finalize_vocab(state, net, vocab, mode_str(mode), req.options).await
}

#[tauri::command]
async fn set_advanced_vocabulary(
    state: State<'_, AppState>,
    req: SetAdvancedVocabularyRequest,
) -> Result<VocabularyInfo, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    if net.kind != kinds::NEXT_TOKEN {
        return Err("vocabularies apply only to next-token networks".to_string());
    }
    let trimmed: Vec<String> = req.tokens.into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if trimmed.is_empty() {
        return Err("advanced vocabulary requires at least one user token".to_string());
    }
    let vocab = Vocabulary::build_advanced(&trimmed);
    finalize_vocab(state, net, vocab, "advanced".into(), VocabularyOptions::default()).await
}

async fn finalize_vocab(
    state: State<'_, AppState>,
    net: Network,
    vocab: Vocabulary,
    mode_label: String,
    options: VocabularyOptions,
) -> Result<VocabularyInfo, String> {
    let tokens = vocab.tokens.clone();
    let info = VocabularyInfo {
        mode: mode_label,
        tokens: tokens.clone(),
        options,
        updated_at: Utc::now(),
    };
    let entry = VocabEntry {
        vocab: Arc::new(RwLock::new(vocab.clone())),
        info: info.clone(),
    };
    state.vocabs.write().await.insert(net.id.clone(), entry);

    // Materialize the model: input = vocab × context, hidden chain, output = vocab.
    rebuild_next_token_model(&state, &net, vocab.size()).await?;
    persist(&state).await?;
    Ok(info)
}

fn mode_str(m: TokenizerMode) -> String {
    match m {
        TokenizerMode::Char     => "char",
        TokenizerMode::Subword  => "subword",
        TokenizerMode::Word     => "word",
        TokenizerMode::Advanced => "advanced",
    }.into()
}

async fn rebuild_next_token_model(
    state: &State<'_, AppState>,
    net: &Network,
    vocab_size: usize,
) -> Result<(), String> {
    let context_size = net.context_size
        .ok_or_else(|| "next-token network missing context_size".to_string())?;
    let hidden = net.hidden_layers.clone().unwrap_or_default();
    let input_dim = vocab_size * context_size;

    // Compose the full layer chain.
    let mut full: Vec<LayerDef> = Vec::with_capacity(hidden.len() + 2);
    // First linear projects input one-hot window into the first hidden width.
    let first_hidden_width = first_linear_in_dim_target(&hidden).unwrap_or(vocab_size);
    full.push(LayerDef::Linear { in_dim: input_dim, out_dim: first_hidden_width });

    // The user's hidden chain starts AT first_hidden_width. The first user
    // Linear (if any) must already have in_dim == first_hidden_width.
    full.extend(hidden.iter().cloned());

    // Output projection to vocab logits.
    let pre_output_dim = running_output_dim(&full, input_dim)?;
    full.push(LayerDef::Linear { in_dim: pre_output_dim, out_dim: vocab_size });

    let (specs, output_dim) = build_layer_specs(&full, input_dim)?;
    let model = Model::from_specs(input_dim, &specs, net.seed);
    let parameter_count = model.parameter_count();

    // Persist the rebuilt model + updated network metadata.
    state.models.write().await.insert(net.id.clone(), Arc::new(RwLock::new(model)));
    if let Some(n) = state.networks.write().await.get_mut(&net.id) {
        n.layers = full;
        n.input_dim = input_dim;
        n.output_dim = output_dim;
        n.parameter_count = parameter_count;
        n.trained = false; // dimensions changed; weights are fresh
    }
    Ok(())
}

/// First Linear's in_dim in the chain, if any.
fn first_linear_in_dim_target(layers: &[LayerDef]) -> Option<usize> {
    for l in layers {
        if let LayerDef::Linear { in_dim, .. } = l { return Some(*in_dim); }
    }
    None
}

/// Output dim of the chain so far, starting from the given input dim.
fn running_output_dim(layers: &[LayerDef], input_dim: usize) -> Result<usize, String> {
    let mut cur = input_dim;
    for (i, l) in layers.iter().enumerate() {
        if let LayerDef::Linear { in_dim, out_dim } = l {
            if *in_dim != cur {
                return Err(format!(
                    "layer {i}: in_dim {in_dim} doesn't match running dim {cur}"
                ));
            }
            cur = *out_dim;
        }
    }
    Ok(cur)
}

#[tauri::command]
async fn get_vocabulary(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<VocabularyInfo>, String> {
    Ok(state.vocabs.read().await.get(&network_id).map(|e| e.info.clone()))
}

#[tauri::command]
async fn tokenize_preview(
    state: State<'_, AppState>,
    network_id: String,
    text: String,
) -> Result<Vec<(u32, String)>, String> {
    let entry = state.vocabs.read().await.get(&network_id).cloned()
        .ok_or_else(|| "no vocabulary on this network".to_string())?;
    let v = entry.vocab.read().await;
    let ids = v.encode(&text);
    Ok(ids.into_iter().map(|id| (id, v.token_of(id).to_string())).collect())
}

// ─── Training ───────────────────────────────────────────────────────────────

#[tauri::command]
async fn start_training(
    app: AppHandle,
    state: State<'_, AppState>,
    req: StartTrainingRequest,
) -> Result<StartTrainingResponse, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let corpus = state.corpora.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "No corpus attached. Add training data on the Corpus tab first.".to_string())?;
    let model_arc = state.models.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| if net.kind == kinds::NEXT_TOKEN {
            "Model not materialized. Build a vocabulary on the Vocabulary tab first.".to_string()
        } else {
            "Model state missing".to_string()
        })?;

    let (x, y, loss_kind, total_examples) = if net.kind == kinds::FEEDFORWARD {
        let ff = corpus.feedforward.as_ref()
            .ok_or_else(|| "Feedforward network requires a feedforward corpus".to_string())?;
        let loss_kind = parse_loss(&req.config.loss)?;
        let x = Tensor::new(vec![ff.rows, ff.in_dim], ff.features.clone());
        let y = Tensor::new(vec![ff.rows, ff.out_dim], ff.targets.clone());
        (x, y, loss_kind, ff.rows)
    } else if net.kind == kinds::NEXT_TOKEN {
        let context_size = net.context_size
            .ok_or_else(|| "Next-token network missing context_size".to_string())?;
        let entry = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing — build one on the Vocabulary tab first".to_string())?;
        let vocab = entry.vocab.read().await.clone();
        let stage = corpus.stage.as_deref().unwrap_or("pretrain");

        let (x, y) = match stage {
            "pretrain" => {
                let text = corpus.text.as_deref()
                    .ok_or_else(|| "Pretraining stage requires a text corpus".to_string())?;
                build_pretraining_tensors(text, &vocab, context_size)
                    .ok_or_else(|| format!(
                        "Corpus has too few tokens for context_size {context_size}"
                    ))?
            }
            "finetune" => {
                let pairs_in = corpus.pairs.as_ref()
                    .ok_or_else(|| "Fine-tuning stage requires input/output pairs".to_string())?;
                if pairs_in.is_empty() {
                    return Err("Fine-tuning needs at least one input/output pair".into());
                }
                let pairs: Vec<Pair> = pairs_in.iter()
                    .map(|p| Pair { input: p.input.clone(), output: p.output.clone() })
                    .collect();
                let mask = req.config.mask_user_tokens.unwrap_or(true);
                build_finetuning_tensors(&pairs, &vocab, context_size, mask)
                    .ok_or_else(|| "Failed to build fine-tuning tensors".to_string())?
            }
            other => return Err(format!("unknown stage '{other}' (expected 'pretrain' or 'finetune')")),
        };
        let n = x.rows();
        (x, y, Loss::CrossEntropy, n)
    } else {
        return Err(format!("unsupported network kind '{}'", net.kind));
    };

    if total_examples == 0 {
        return Err("No training examples produced from the current corpus".into());
    }
    if req.config.epochs == 0 {
        return Err("epochs must be > 0".into());
    }
    // Refuse to "train" a network with no parameters — that's how next-token
    // networks end up stuck on "epoch 0 / 500" forever when the vocabulary
    // wasn't built. The model has nothing to learn, but the loop runs anyway.
    if net.parameter_count == 0 {
        return Err(
            "Network has no parameters yet. For next-token networks, set a \
             corpus (which builds the vocabulary and materialises the model) \
             before training."
                .into(),
        );
    }
    let batch_size = req.config.batch_size.max(1);

    let training_id = Uuid::new_v4().to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    let rollback = Arc::new(AtomicBool::new(false));
    let training_state = Arc::new(RwLock::new(TrainingState {
        running: true,
        total_epochs: req.config.epochs,
        ..Default::default()
    }));
    state.trainers.write().await.insert(training_id.clone(), TrainerHandle {
        state: training_state.clone(),
        cancel: cancel.clone(),
        rollback: rollback.clone(),
    });

    let networks_handle = state.networks.clone();
    let id_clone = training_id.clone();
    let cfg = req.config.clone();
    let net_id = req.network_id.clone();

    tokio::spawn(async move {
        run_training_loop(
            app, id_clone, net_id, cfg, model_arc, networks_handle,
            x, y, loss_kind, batch_size, training_state, cancel, rollback,
        ).await;
        // The TrainerHandle stays in `state.trainers` so callers can still
        // query its final status. Map cleanup happens at app shutdown.
    });

    Ok(StartTrainingResponse { training_id, status: "running".into() })
}

#[tauri::command]
async fn stop_training(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<bool, String> {
    let trainers = state.trainers.read().await;
    let Some(handle) = trainers.get(&training_id).cloned() else { return Ok(false); };
    handle.cancel.store(true, Ordering::Relaxed);
    Ok(true)
}

/// Cancel training AND roll back to the pre-training weights. Used by the
/// "Abort" button — `stop_training` keeps whatever weights training has
/// produced so far, `abort_training` discards them.
#[tauri::command]
async fn abort_training(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<bool, String> {
    let trainers = state.trainers.read().await;
    let Some(handle) = trainers.get(&training_id).cloned() else { return Ok(false); };
    handle.rollback.store(true, Ordering::Relaxed);
    handle.cancel.store(true, Ordering::Relaxed);
    Ok(true)
}

#[tauri::command]
async fn get_training_status(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<TrainingStatusResponse, String> {
    let trainers = state.trainers.read().await;
    let h = trainers.get(&training_id)
        .ok_or_else(|| "Training session not found".to_string())?
        .clone();
    drop(trainers);
    let s = h.state.read().await;
    let status = if s.error.is_some() { "error" }
                 else if s.cancelled  { "cancelled" }
                 else if s.stopped    { "completed" }
                 else                 { "running" };
    Ok(TrainingStatusResponse {
        training_id,
        status: status.into(),
        epoch: s.epoch,
        total_epochs: s.total_epochs,
        last_loss: s.last_loss,
        loss_history: s.loss_history.clone(),
        elapsed_secs: s.elapsed_secs,
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_training_loop(
    app: AppHandle,
    training_id: String,
    network_id: String,
    cfg: TrainingConfig,
    model_arc: Arc<RwLock<Model>>,
    networks_handle: Arc<RwLock<HashMap<String, Network>>>,
    x: Tensor,
    y: Tensor,
    loss_kind: Loss,
    batch_size: usize,
    training_state: Arc<RwLock<TrainingState>>,
    cancel: Arc<AtomicBool>,
    rollback: Arc<AtomicBool>,
) {
    let n = x.rows();
    let in_dim = x.cols();
    let out_dim = y.cols();

    // Snapshot the pre-training weights so that "Abort" can restore them. This
    // is the only point at which we keep a long-lived clone of the model.
    let initial_snapshot: Model = {
        let m = model_arc.read().await;
        m.clone()
    };

    // Build the optimizer from the model's parameter shapes. We acquire the
    // lock briefly here (not held across the training loop) so that other
    // commands — persistence, inference — can still access the model.
    let shapes = {
        let m = model_arc.read().await;
        m.parameter_shapes()
    };
    let mut optimizer = match build_optimizer(&cfg.optimizer, &shapes) {
        Ok(o) => o,
        Err(e) => {
            mark_error(&training_state, &training_id, &app, e).await;
            return;
        }
    };

    let start = std::time::Instant::now();
    let mut loss_history: Vec<f32> = Vec::with_capacity(cfg.epochs);
    let mut rng = SplitMix64::new(cfg.seed.wrapping_add(0xA5A5_5A5A_5A5A_A5A5));
    let mut indices: Vec<usize> = (0..n).collect();

    /// Run the cleanup path for a cancelled training run. Honours `rollback`
    /// to decide whether to keep or revert the in-progress weights, persists
    /// the result, and emits the `training_finished` event.
    async fn finish_cancelled(
        app: &AppHandle,
        training_id: &str,
        cfg_epochs: usize,
        loss_history: &[f32],
        start: std::time::Instant,
        training_state: &Arc<RwLock<TrainingState>>,
        model_arc: &Arc<RwLock<Model>>,
        rollback: &Arc<AtomicBool>,
        initial_snapshot: &Model,
    ) {
        if rollback.load(Ordering::Relaxed) {
            // Restore the model to its pre-training state. Any progress made
            // during this run is discarded — that's the whole point of Abort.
            let mut m = model_arc.write().await;
            *m = initial_snapshot.clone();
        }
        let elapsed = start.elapsed().as_secs_f32();
        let final_loss = loss_history.last().copied().unwrap_or(0.0);
        {
            let mut s = training_state.write().await;
            s.running = false; s.stopped = true; s.cancelled = true;
            s.elapsed_secs = elapsed;
        }
        if let Some(app_state) = app.try_state::<AppState>() {
            persistence::save_best_effort(app_state.inner()).await;
        }
        let _ = app.emit("training_finished", TrainingFinished {
            training_id: training_id.to_string(),
            status: if rollback.load(Ordering::Relaxed) { "aborted".into() }
                    else { "cancelled".into() },
            final_loss,
            total_epochs: cfg_epochs,
            elapsed_secs: elapsed,
        });
    }

    for epoch in 1..=cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            finish_cancelled(
                &app, &training_id, cfg.epochs, &loss_history, start,
                &training_state, &model_arc, &rollback, &initial_snapshot,
            ).await;
            return;
        }

        // Shuffle each epoch (deterministic from seed).
        for i in (1..indices.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0_f32;
        let mut batches = 0usize;
        for chunk in indices.chunks(batch_size) {
            if cancel.load(Ordering::Relaxed) { break; }
            let mut bx = Vec::with_capacity(chunk.len() * in_dim);
            let mut by = Vec::with_capacity(chunk.len() * out_dim);
            for &i in chunk {
                bx.extend_from_slice(&x.data[i * in_dim..(i + 1) * in_dim]);
                by.extend_from_slice(&y.data[i * out_dim..(i + 1) * out_dim]);
            }
            let bx = Tensor::new(vec![chunk.len(), in_dim], bx);
            let by = Tensor::new(vec![chunk.len(), out_dim], by);
            // Acquire the model lock for just one optimisation step, then
            // release it. This is what lets `persist()` and `infer()` proceed
            // between batches instead of waiting for the whole run to finish.
            let loss = {
                let mut model = model_arc.write().await;
                model.train_step(&mut optimizer, loss_kind, &bx, &by)
            };
            if !loss.is_finite() {
                mark_error(&training_state, &training_id, &app,
                    format!("Loss diverged to {loss} at epoch {epoch}")).await;
                return;
            }
            epoch_loss += loss;
            batches += 1;
        }
        let mean_loss = epoch_loss / batches.max(1) as f32;
        loss_history.push(mean_loss);

        let elapsed = start.elapsed().as_secs_f32();
        {
            let mut s = training_state.write().await;
            s.epoch = epoch;
            s.last_loss = mean_loss;
            s.loss_history = loss_history.clone();
            s.elapsed_secs = elapsed;
        }
        let _ = app.emit("training_update", TrainingUpdate {
            training_id: training_id.clone(),
            epoch,
            total_epochs: cfg.epochs,
            loss: mean_loss,
            loss_history: loss_history.clone(),
            elapsed_secs: elapsed,
        });

        tokio::task::yield_now().await;
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    {
        let mut s = training_state.write().await;
        s.running = false; s.stopped = true;
        s.elapsed_secs = elapsed;
    }
    if let Some(net) = networks_handle.write().await.get_mut(&network_id) {
        net.trained = true;
    }
    // Persist the freshly-trained weights and the `trained = true` flag. We
    // pull AppState off the AppHandle since the training loop runs on a
    // detached task and doesn't carry the State<'_, AppState> guard.
    if let Some(app_state) = app.try_state::<AppState>() {
        persistence::save_best_effort(app_state.inner()).await;
    }
    let _ = app.emit("training_finished", TrainingFinished {
        training_id,
        status: "completed".into(),
        final_loss,
        total_epochs: cfg.epochs,
        elapsed_secs: elapsed,
    });
}

async fn mark_error(
    state: &Arc<RwLock<TrainingState>>,
    training_id: &str,
    app: &AppHandle,
    msg: String,
) {
    {
        let mut s = state.write().await;
        s.error = Some(msg.clone());
        s.running = false; s.stopped = true;
    }
    let _ = app.emit("training_error", TrainingError {
        training_id: training_id.to_string(),
        message: msg,
    });
}

// ─── Inference ──────────────────────────────────────────────────────────────

#[tauri::command]
async fn infer(
    app: AppHandle,
    state: State<'_, AppState>,
    req: InferRequest,
) -> Result<InferResponse, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let model_arc = state.models.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| if net.kind == kinds::NEXT_TOKEN {
            "Model not materialized. Build a vocabulary on the Vocabulary tab first.".to_string()
        } else {
            "Model state missing".to_string()
        })?;

    if net.kind == kinds::FEEDFORWARD {
        let model = model_arc.read().await;
        let features = req.features
            .ok_or_else(|| "features array required for feedforward inference".to_string())?;
        if features.len() != net.input_dim {
            return Err(format!(
                "features length {} doesn't match input_dim {}",
                features.len(), net.input_dim
            ));
        }
        let x = Tensor::new(vec![1, net.input_dim], features);
        let y = model.predict(&x);
        Ok(InferResponse { output: Some(y.data), inference_id: None })
    } else if net.kind == kinds::NEXT_TOKEN {
        let prompt = req.prompt.unwrap_or_default();
        let max_new = req.max_new_tokens.unwrap_or(64).min(4096).max(1);
        let temperature = req.temperature.unwrap_or(0.0).max(0.0);
        let context_size = net.context_size
            .ok_or_else(|| "Next-token network missing context_size".to_string())?;
        let entry = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing — build one on the Vocabulary tab first".to_string())?;

        let inference_id = Uuid::new_v4().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        state.inferrers.write().await.insert(inference_id.clone(), InferenceHandle {
            cancel: cancel.clone(),
        });

        let inferrers_handle = state.inferrers.clone();
        let id_clone = inference_id.clone();
        tokio::spawn(async move {
            run_generation(
                app, id_clone.clone(), model_arc, entry.vocab,
                prompt, context_size, max_new, temperature, cancel,
            ).await;
            inferrers_handle.write().await.remove(&id_clone);
        });

        Ok(InferResponse { output: None, inference_id: Some(inference_id) })
    } else {
        Err(format!("unsupported network kind '{}'", net.kind))
    }
}

#[tauri::command]
async fn stop_inference(
    state: State<'_, AppState>,
    inference_id: String,
) -> Result<bool, String> {
    let h = state.inferrers.read().await.get(&inference_id).cloned();
    if let Some(h) = h {
        h.cancel.store(true, Ordering::Relaxed);
        Ok(true)
    } else { Ok(false) }
}

#[allow(clippy::too_many_arguments)]
async fn run_generation(
    app: AppHandle,
    inference_id: String,
    model_arc: Arc<RwLock<Model>>,
    vocab_arc: Arc<RwLock<Vocabulary>>,
    prompt: String,
    context_size: usize,
    max_new_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) {
    let model  = model_arc.read().await;
    let vocab  = vocab_arc.read().await;
    let mut ids: Vec<u32> = vocab.encode(&prompt);
    let initial_len = ids.len();
    let mut generated = String::new();
    let mut rng = SplitMix64::new(0xDEAD_BEEF_u64 ^ (max_new_tokens as u64).wrapping_mul(0x9E3779B1));

    for index in 0..max_new_tokens {
        if cancel.load(Ordering::Relaxed) {
            let _ = app.emit("inference_finished", InferenceFinished {
                inference_id: inference_id.clone(),
                status: "cancelled".into(),
                generated: generated.clone(),
                token_count: ids.len() - initial_len,
            });
            return;
        }

        let ctx = encode_context(&ids, &vocab, context_size);
        let logits = model.predict(&ctx);
        let probs = softmax_rows(&logits);
        let row = &probs.data[..probs.cols()];

        let chosen = if temperature <= f32::EPSILON {
            argmax(row)
        } else {
            sample_with_temperature(row, temperature, &mut rng)
        };
        let prob = row[chosen];
        let token = vocab.token_of(chosen as u32).to_string();

        // Emit one token at a time for streaming UI.
        let visible_token = if (chosen as u32) < neuralcabin_engine::tokenizer::RESERVED as u32 {
            // Don't append reserved tokens to generated text, but still report them.
            String::new()
        } else {
            token.clone()
        };
        generated.push_str(&visible_token);

        let _ = app.emit("inference_token", InferenceToken {
            inference_id: inference_id.clone(),
            index, token: token.clone(), probability: prob,
        });

        if (chosen as u32) == EOS_ID { break; }
        ids.push(chosen as u32);

        // Cooperative yield so the cancel flag can be observed promptly.
        tokio::task::yield_now().await;
    }

    let _ = app.emit("inference_finished", InferenceFinished {
        inference_id: inference_id.clone(),
        status: "completed".into(),
        generated,
        token_count: ids.len() - initial_len,
    });
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv { bv = v; best = i; }
    }
    best
}

fn sample_with_temperature(probs: &[f32], temperature: f32, rng: &mut SplitMix64) -> usize {
    let mut logits: Vec<f32> = probs.iter().map(|p| p.max(1e-12).ln() / temperature).collect();
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
    for v in logits.iter_mut() { *v /= sum; }

    let r = (rng.next_u64() as f64 / u64::MAX as f64) as f32;
    let mut acc = 0.0_f32;
    for (i, &p) in logits.iter().enumerate() {
        acc += p;
        if r <= acc { return i; }
    }
    logits.len() - 1
}

// ─── Tauri entry point ──────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new();
    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            // Resolve the per-user data directory and load any saved workspace
            // synchronously before the frontend can issue its first command —
            // otherwise `list_networks` would briefly return an empty list and
            // a fast user could overwrite an existing save.
            let data_dir = app.path().app_data_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("neuralcabin"));
            let state: State<'_, AppState> = app.state();
            let app_state: &AppState = state.inner();
            tauri::async_runtime::block_on(async {
                *app_state.data_dir.write().await = Some(data_dir.clone());
                match persistence::load_from_dir(&data_dir).await {
                    Ok(Some(persisted)) => {
                        persistence::apply(app_state, persisted).await;
                        eprintln!("[neuralcabin] loaded state from {}", data_dir.display());
                    }
                    Ok(None) => {
                        eprintln!("[neuralcabin] no saved state at {} — starting fresh", data_dir.display());
                    }
                    Err(e) => {
                        // Corrupt file: surface loudly. We don't delete it so
                        // the user can recover by editing or removing it.
                        eprintln!("[neuralcabin] ERROR loading state from {}: {e}", data_dir.display());
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            create_network, list_networks, get_network, delete_network,
            set_corpus, get_corpus, corpus_stats,
            build_vocabulary, set_advanced_vocabulary, get_vocabulary, tokenize_preview,
            start_training, stop_training, abort_training, get_training_status,
            infer, stop_inference,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

