//! Background text-generation thread.
//!
//! Works like the trainer: the UI spawns a [`GeneratorHandle`], polls it each
//! frame for new token strings, and drops the handle to stop generation.

use crate::corpus::{positional_enc_pub, random_embedding_table};
use crate::networks::EmbeddingKind;
use crate::vocab::VocabMode;
use neuralcabin_engine::tensor::{SplitMix64, Tensor};
use neuralcabin_engine::Model;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};

/// Everything the background thread needs — all owned, no lifetime issues.
pub struct GeneratorConfig {
    pub model: Model,
    pub vocab_tokens: Vec<String>,
    pub vocab_mode: VocabMode,
    pub embedding: EmbeddingKind,
    pub embed_dim: usize,
    pub context_size: usize,
    pub seed: u64,
    pub temperature: f32,
    pub max_tokens: usize,
    /// IDF weights (one per vocab token). Ones for non-TfIdf modes.
    pub idf: Vec<f32>,
    /// Pre-encoded prompt token ids. Seeded with `[1]` if the prompt was empty.
    pub history: Vec<usize>,
    /// True when the encoded prompt was non-empty (affects leading-space logic
    /// in word mode).
    pub prompt_nonempty: bool,
}

pub struct GeneratorHandle {
    /// Decoded token pieces arrive here one by one.
    pub token_rx: Receiver<String>,
    stop_tx: SyncSender<()>,
    join: Option<JoinHandle<()>>,
}

impl GeneratorHandle {
    /// Collect all token strings that have arrived since the last call.
    pub fn drain(&self) -> Vec<String> {
        self.token_rx.try_iter().collect()
    }

    pub fn is_finished(&self) -> bool {
        self.join.as_ref().is_none_or(|j| j.is_finished())
    }
}

/// Sending a stop signal on drop ensures the thread shuts down even when the
/// handle is simply discarded (e.g. real-time mode restarts generation).
impl Drop for GeneratorHandle {
    fn drop(&mut self) {
        let _ = self.stop_tx.try_send(());
    }
}

pub fn spawn(cfg: GeneratorConfig) -> GeneratorHandle {
    let (token_tx, token_rx) = mpsc::channel::<String>();
    let (stop_tx, stop_rx) = mpsc::sync_channel::<()>(1);
    let join = thread::Builder::new()
        .name("neuralcabin-generator".into())
        .spawn(move || run(cfg, token_tx, stop_rx))
        .expect("spawn generator thread");
    GeneratorHandle { token_rx, stop_tx, join: Some(join) }
}

fn run(cfg: GeneratorConfig, token_tx: mpsc::Sender<String>, stop_rx: Receiver<()>) {
    let v = cfg.vocab_tokens.len().max(1);
    let ctx = cfg.context_size.max(1);
    let e = cfg.embed_dim.max(1);
    let expected_in = cfg.embedding.input_dim(ctx, v, e);
    let emb_table = cfg.embedding.uses_dense_embed()
        .then(|| random_embedding_table(v, e, cfg.seed));
    let word_mode = cfg.vocab_mode == VocabMode::Word;
    let temp = cfg.temperature.max(1e-3);
    let mut history = cfg.history;
    let mut rng = SplitMix64::new(
        cfg.seed.wrapping_add(history.len() as u64).wrapping_add(0xBEEF),
    );

    for token_count in 0..cfg.max_tokens {
        // Check for stop before each token — keeps latency short.
        match stop_rx.try_recv() {
            Ok(()) | Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }

        let input_vec = build_input_vec(
            &history, ctx, v, e, cfg.embedding, emb_table.as_deref(), &cfg.idf,
        );
        let logits = cfg.model.predict(&Tensor::new(vec![1, expected_in], input_vec));

        // Softmax with temperature.
        let mut row: Vec<f32> = logits.data.iter().map(|x| x / temp).collect();
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for r in row.iter_mut() { *r = (*r - max).exp(); }
        let sum: f32 = row.iter().sum();
        if sum <= 0.0 || !sum.is_finite() { break; }
        for r in row.iter_mut() { *r /= sum; }

        // Sample.
        let u = rng.next_f32();
        let mut acc = 0.0_f32;
        let mut chosen = 0usize;
        for (i, p) in row.iter().enumerate() {
            acc += *p;
            if u <= acc { chosen = i; break; }
        }

        if chosen == 0 { break; } // <unk> = EOS

        let tok = cfg.vocab_tokens[chosen].as_str();
        let piece = if word_mode && (token_count > 0 || cfg.prompt_nonempty) {
            format!(" {tok}")
        } else {
            tok.to_owned()
        };

        if token_tx.send(piece).is_err() { break; } // receiver dropped
        history.push(chosen);
    }
}

fn build_input_vec(
    history: &[usize],
    ctx: usize,
    v: usize,
    edim: usize,
    emb: EmbeddingKind,
    emb_table: Option<&[f32]>,
    idf: &[f32],
) -> Vec<f32> {
    match emb {
        EmbeddingKind::OneHot | EmbeddingKind::TfIdf => {
            let mut input = vec![0.0_f32; ctx * v];
            for k in 0..ctx {
                let idx = history.len().checked_sub(ctx - k).map(|i| history[i]).unwrap_or(0);
                let id = idx.min(v - 1);
                input[k * v + id] =
                    if emb == EmbeddingKind::TfIdf { idf[id] } else { 1.0 };
            }
            input
        }
        EmbeddingKind::FastText | EmbeddingKind::Transformer => {
            let e = edim.max(1);
            let table = emb_table.unwrap_or(&[]);
            let mut input = vec![0.0_f32; ctx * e];
            for k in 0..ctx {
                let idx = history.len().checked_sub(ctx - k).map(|i| history[i]).unwrap_or(0);
                let id = idx.min(v - 1);
                let row_start = id * e;
                for d in 0..e {
                    let mut val = if row_start + d < table.len() { table[row_start + d] } else { 0.0 };
                    if emb == EmbeddingKind::Transformer { val += positional_enc_pub(k, d, e); }
                    input[k * e + d] = val;
                }
            }
            input
        }
    }
}
