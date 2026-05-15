mod models;

use models::*;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;
use chrono::Utc;
use neuralcabin_engine::nn::{LayerSpec, Model};
use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
use neuralcabin_engine::tensor::{SplitMix64, Tensor};
use neuralcabin_engine::tokenizer::{TokenizerMode, Vocabulary};
use neuralcabin_engine::corpus::{
    build_finetuning_tensors, build_pretraining_tensors, encode_context, Pair,
};
use neuralcabin_engine::activations::softmax_rows;
use neuralcabin_engine::{Activation, Loss};

// ─── State ──────────────────────────────────────────────────────────────────

pub struct AppState {
    networks: Arc<RwLock<HashMap<String, Network>>>,
    /// Trained model weights, keyed by network_id.
    models:   Arc<RwLock<HashMap<String, Arc<RwLock<Model>>>>>,
    /// Vocabularies for next-token networks, keyed by network_id.
    vocabs:   Arc<RwLock<HashMap<String, Arc<RwLock<Vocabulary>>>>>,
    /// Corpus per network.
    corpora:  Arc<RwLock<HashMap<String, Corpus>>>,
    /// Active training sessions.
    trainers: Arc<RwLock<HashMap<String, Arc<RwLock<TrainingState>>>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            networks: Arc::new(RwLock::new(HashMap::new())),
            models:   Arc::new(RwLock::new(HashMap::new())),
            vocabs:   Arc::new(RwLock::new(HashMap::new())),
            corpora:  Arc::new(RwLock::new(HashMap::new())),
            trainers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

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
        "char" => Ok(TokenizerMode::Char),
        "word" => Ok(TokenizerMode::Word),
        other  => Err(format!("unknown vocab mode '{other}' (expected 'char' or 'word')")),
    }
}

fn build_optimizer(cfg: &OptimizerConfig, shapes: &[Vec<usize>]) -> Result<Optimizer, String> {
    let kind = match cfg.kind.to_ascii_lowercase().as_str() {
        "adam" => OptimizerKind::Adam {
            lr:    cfg.lr,
            beta1: cfg.beta1.unwrap_or(0.9),
            beta2: cfg.beta2.unwrap_or(0.999),
            eps:   cfg.eps.unwrap_or(1e-8),
        },
        "sgd" => OptimizerKind::Sgd {
            lr:       cfg.lr,
            momentum: cfg.momentum.unwrap_or(0.0),
        },
        other => return Err(format!("unknown optimizer '{other}' (expected 'adam' or 'sgd')")),
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
    if req.layers.is_empty() {
        return Err("network must have at least one layer".to_string());
    }
    if req.input_dim == 0 {
        return Err("input_dim must be > 0".to_string());
    }

    let (specs, output_dim) = build_layer_specs(&req.layers, req.input_dim)?;
    let model = Model::from_specs(req.input_dim, &specs, req.seed);
    let parameter_count = model.parameter_count();

    let id = Uuid::new_v4().to_string();
    let network = Network {
        id: id.clone(),
        name: req.name,
        kind: req.kind,
        layers: req.layers,
        seed: req.seed,
        input_dim: req.input_dim,
        output_dim,
        created_at: Utc::now(),
        parameter_count,
        trained: false,
        context_size: req.context_size,
    };

    state.networks.write().await.insert(id.clone(), network.clone());
    state.models.write().await.insert(id, Arc::new(RwLock::new(model)));
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
    Ok(state.networks.write().await.remove(&id).is_some())
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
        vocab_mode: None,
        vocab: None,
        stage: None,
    };

    if net.kind == kinds::FEEDFORWARD {
        let ff = req.feedforward.ok_or_else(|| "feedforward corpus required".to_string())?;
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
            return Err("feedforward features length doesn't match rows × in_dim".to_string());
        }
        if ff.targets.len() != ff.rows * ff.out_dim {
            return Err("feedforward targets length doesn't match rows × out_dim".to_string());
        }
        if ff.rows == 0 {
            return Err("feedforward corpus must have at least one row".to_string());
        }
        corpus.feedforward = Some(ff);
    } else if net.kind == kinds::NEXT_TOKEN {
        let mode = parse_tokenizer_mode(req.vocab_mode.as_deref().unwrap_or("char"))?;
        corpus.vocab_mode = Some(mode.name().to_ascii_lowercase());
        corpus.text  = req.text.clone();
        corpus.pairs = req.pairs.clone();
        corpus.stage = req.stage.clone();

        let mut owned_strings: Vec<String> = Vec::new();
        if let Some(t) = req.text.as_ref() { owned_strings.push(t.clone()); }
        if let Some(ps) = req.pairs.as_ref() {
            for p in ps {
                owned_strings.push(p.input.clone());
                owned_strings.push(p.output.clone());
            }
        }
        let refs: Vec<&str> = owned_strings.iter().map(|s| s.as_str()).collect();
        let vocab = Vocabulary::from_corpus(mode, &refs);
        corpus.vocab = Some(vocab.tokens.clone());
        state.vocabs.write().await.insert(net.id.clone(), Arc::new(RwLock::new(vocab)));
    } else {
        return Err(format!("unsupported network kind '{}'", net.kind));
    }

    let vocabs_guard = state.vocabs.read().await;
    let stats = compute_stats(&net, &corpus, &vocabs_guard).await;
    drop(vocabs_guard);
    state.corpora.write().await.insert(net.id, corpus);
    Ok(stats)
}

async fn compute_stats(
    net: &Network,
    corpus: &Corpus,
    vocabs: &HashMap<String, Arc<RwLock<Vocabulary>>>,
) -> CorpusStats {
    let mut stats = CorpusStats {
        kind: net.kind.clone(),
        stage: corpus.stage.clone(),
        rows: None, in_dim: None, out_dim: None,
        text_chars: None, text_tokens: None, pair_count: None,
        vocab_size: None, vocab_mode: corpus.vocab_mode.clone(),
        training_examples: None,
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
        if let Some(vocab) = vocabs.get(&net.id) {
            let v = vocab.read().await;
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
    let net = match state.networks.read().await.get(&network_id).cloned() {
        Some(n) => n,
        None => return Err("Network not found".into()),
    };
    let corpora = state.corpora.read().await;
    let Some(corpus) = corpora.get(&network_id) else { return Ok(None); };
    let vocabs = state.vocabs.read().await;
    let stats = compute_stats(&net, corpus, &vocabs).await;
    Ok(Some(stats))
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
        .ok_or_else(|| "Model state missing".to_string())?;

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
        let vocab_arc = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing — set the corpus first".to_string())?;
        let vocab = vocab_arc.read().await.clone();
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
    let batch_size = req.config.batch_size.max(1);

    let training_id = Uuid::new_v4().to_string();
    let training_state = Arc::new(RwLock::new(TrainingState {
        running: true,
        total_epochs: req.config.epochs,
        ..Default::default()
    }));
    state.trainers.write().await.insert(training_id.clone(), training_state.clone());

    let networks_handle = state.networks.clone();
    let id_clone = training_id.clone();
    let cfg = req.config.clone();
    let net_id = req.network_id.clone();

    tokio::spawn(async move {
        run_training_loop(
            app, id_clone, net_id, cfg, model_arc, networks_handle,
            x, y, loss_kind, batch_size, training_state,
        ).await;
    });

    Ok(StartTrainingResponse { training_id, status: "running".into() })
}

#[tauri::command]
async fn get_training_status(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<TrainingStatusResponse, String> {
    let trainers = state.trainers.read().await;
    let ts = trainers.get(&training_id)
        .ok_or_else(|| "Training session not found".to_string())?
        .clone();
    drop(trainers);
    let s = ts.read().await;
    let status = if s.error.is_some() { "error" }
                 else if s.stopped     { "completed" }
                 else                  { "running" };
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
) {
    let n = x.rows();
    let in_dim = x.cols();
    let out_dim = y.cols();
    let mut model = model_arc.write().await;
    let shapes = model.parameter_shapes();
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

    for epoch in 1..=cfg.epochs {
        for i in (1..indices.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0_f32;
        let mut batches = 0usize;
        for chunk in indices.chunks(batch_size) {
            let mut bx = Vec::with_capacity(chunk.len() * in_dim);
            let mut by = Vec::with_capacity(chunk.len() * out_dim);
            for &i in chunk {
                bx.extend_from_slice(&x.data[i * in_dim..(i + 1) * in_dim]);
                by.extend_from_slice(&y.data[i * out_dim..(i + 1) * out_dim]);
            }
            let bx = Tensor::new(vec![chunk.len(), in_dim], bx);
            let by = Tensor::new(vec![chunk.len(), out_dim], by);
            let loss = model.train_step(&mut optimizer, loss_kind, &bx, &by);
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
        s.running = false;
        s.stopped = true;
        s.elapsed_secs = elapsed;
    }
    if let Some(net) = networks_handle.write().await.get_mut(&network_id) {
        net.trained = true;
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
        s.running = false;
        s.stopped = true;
    }
    let _ = app.emit("training_error", TrainingError {
        training_id: training_id.to_string(),
        message: msg,
    });
}

// ─── Inference ──────────────────────────────────────────────────────────────

#[tauri::command]
async fn infer(
    state: State<'_, AppState>,
    req: InferRequest,
) -> Result<InferResponse, String> {
    let net = state.networks.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Network not found".to_string())?;
    let model_arc = state.models.read().await.get(&req.network_id).cloned()
        .ok_or_else(|| "Model state missing".to_string())?;
    let model = model_arc.read().await;

    if net.kind == kinds::FEEDFORWARD {
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
        Ok(InferResponse { output: Some(y.data), generated: None, steps: None })
    } else if net.kind == kinds::NEXT_TOKEN {
        let prompt = req.prompt.unwrap_or_default();
        let max_new = req.max_new_tokens.unwrap_or(64).min(2048);
        let temperature = req.temperature.unwrap_or(0.0).max(0.0);
        let context_size = net.context_size
            .ok_or_else(|| "Next-token network missing context_size".to_string())?;
        let vocab_arc = state.vocabs.read().await.get(&req.network_id).cloned()
            .ok_or_else(|| "Vocabulary missing — set the corpus first".to_string())?;
        let vocab = vocab_arc.read().await.clone();

        let mut ids: Vec<u32> = vocab.encode(&prompt);
        let initial_len = ids.len();
        let mut steps = Vec::with_capacity(max_new);
        let mut rng = SplitMix64::new(0xDEAD_BEEFu64.wrapping_add(max_new as u64));

        for _ in 0..max_new {
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
            steps.push(GenerationStep { token: token.clone(), probability: prob });

            if (chosen as u32) == neuralcabin_engine::tokenizer::EOS_ID { break; }
            ids.push(chosen as u32);
        }

        let generated_ids: Vec<u32> = ids.iter().skip(initial_len).copied().collect();
        let generated = vocab.decode(&generated_ids);
        Ok(InferResponse { output: None, generated: Some(generated), steps: Some(steps) })
    } else {
        Err(format!("unsupported network kind '{}'", net.kind))
    }
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

// ─── Vocabulary commands ────────────────────────────────────────────────────

#[tauri::command]
async fn get_vocabulary(
    state: State<'_, AppState>,
    network_id: String,
) -> Result<Option<Vec<String>>, String> {
    let vocabs = state.vocabs.read().await;
    if let Some(v) = vocabs.get(&network_id) {
        let g = v.read().await;
        Ok(Some(g.tokens.clone()))
    } else {
        Ok(None)
    }
}

// ─── Tauri entry point ──────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            create_network, list_networks, get_network, delete_network,
            set_corpus, get_corpus, corpus_stats,
            get_vocabulary,
            start_training, get_training_status,
            infer,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
