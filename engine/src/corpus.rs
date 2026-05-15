//! Convert a text corpus into a real next-token-prediction training set.
//!
//! Strategy: sliding windows of `context_size` tokens predict the next token.
//! Inputs are flattened one-hot vectors of length `context_size * vocab_size`.
//! Targets are one-hot vectors of length `vocab_size`. This works directly with
//! the existing Linear + Activation layer stack — no new layer types required.
//!
//! For fine-tuning (input/output pairs), the same shape applies: each pair is
//! tokenized into a single sequence `<bos> input <sep> output <eos>` and we
//! emit one training example per output token.

use crate::tensor::Tensor;
use crate::tokenizer::{Vocabulary, BOS_ID, EOS_ID, PAD_ID};

/// A single fine-tuning conversation pair.
#[derive(Clone, Debug)]
pub struct Pair {
    pub input: String,
    pub output: String,
}

/// Build (X, Y) tensors from raw text using sliding windows.
///
/// `X.shape == [n_examples, context_size * vocab_size]`
/// `Y.shape == [n_examples, vocab_size]`
///
/// Returns `None` if the corpus has fewer than `context_size + 1` tokens.
pub fn build_pretraining_tensors(
    text: &str,
    vocab: &Vocabulary,
    context_size: usize,
) -> Option<(Tensor, Tensor)> {
    assert!(context_size > 0, "context_size must be positive");
    let ids = vocab.encode(text);
    if ids.len() < context_size + 1 { return None; }

    let v = vocab.size();
    let n_examples = ids.len() - context_size;

    let mut x = vec![0.0_f32; n_examples * context_size * v];
    let mut y = vec![0.0_f32; n_examples * v];

    for i in 0..n_examples {
        let window = &ids[i..i + context_size];
        let target = ids[i + context_size];
        for (pos, &tok) in window.iter().enumerate() {
            x[i * context_size * v + pos * v + tok as usize] = 1.0;
        }
        y[i * v + target as usize] = 1.0;
    }

    Some((
        Tensor::new(vec![n_examples, context_size * v], x),
        Tensor::new(vec![n_examples, v], y),
    ))
}

/// Build (X, Y) tensors from input/output pairs for fine-tuning.
///
/// Each pair is encoded as `<bos> input <eos> output <eos>`. Sliding windows
/// then teach the model to predict each token of the output (and the trailing
/// <eos>) given the prior tokens.
///
/// If `mask_user_tokens` is true, only windows whose target falls strictly in
/// the output region are emitted — the model is not penalised for failing to
/// reproduce the user's input.
pub fn build_finetuning_tensors(
    pairs: &[Pair],
    vocab: &Vocabulary,
    context_size: usize,
    mask_user_tokens: bool,
) -> Option<(Tensor, Tensor)> {
    assert!(context_size > 0, "context_size must be positive");
    let v = vocab.size();
    let mut x_rows: Vec<Vec<f32>> = Vec::new();
    let mut y_rows: Vec<u32> = Vec::new();

    for pair in pairs {
        let mut seq: Vec<u32> = Vec::new();
        seq.push(BOS_ID);
        seq.extend(vocab.encode(&pair.input));
        seq.push(EOS_ID);
        let output_start = seq.len();
        seq.extend(vocab.encode(&pair.output));
        seq.push(EOS_ID);

        if seq.len() < 2 { continue; }

        for i in 0..(seq.len() - 1) {
            let target = seq[i + 1];
            if mask_user_tokens && (i + 1) < output_start { continue; }

            // Build context: last `context_size` tokens up to and including `i`,
            // left-padded with PAD if shorter.
            let mut window = vec![0.0_f32; context_size * v];
            let start = if i + 1 > context_size { i + 1 - context_size } else { 0 };
            let prefix_len = (i + 1) - start;
            let pad_len = context_size - prefix_len;
            for p in 0..pad_len {
                window[p * v + PAD_ID as usize] = 1.0;
            }
            for (offset, &tok) in seq[start..=i].iter().enumerate() {
                let pos = pad_len + offset;
                window[pos * v + tok as usize] = 1.0;
            }
            x_rows.push(window);
            y_rows.push(target);
        }
    }

    if x_rows.is_empty() { return None; }

    let n = x_rows.len();
    let mut x_flat = Vec::with_capacity(n * context_size * v);
    for row in &x_rows { x_flat.extend_from_slice(row); }
    let mut y_flat = vec![0.0_f32; n * v];
    for (i, &t) in y_rows.iter().enumerate() {
        y_flat[i * v + t as usize] = 1.0;
    }

    Some((
        Tensor::new(vec![n, context_size * v], x_flat),
        Tensor::new(vec![n, v], y_flat),
    ))
}

/// Encode a single context window for inference. Returns a `[1, context_size * vocab_size]`
/// tensor with the most recent `context_size` tokens (left-padded with PAD).
pub fn encode_context(ids: &[u32], vocab: &Vocabulary, context_size: usize) -> Tensor {
    let v = vocab.size();
    let mut window = vec![0.0_f32; context_size * v];
    let start = if ids.len() > context_size { ids.len() - context_size } else { 0 };
    let prefix_len = ids.len() - start;
    let pad_len = context_size - prefix_len;
    for p in 0..pad_len {
        window[p * v + PAD_ID as usize] = 1.0;
    }
    for (offset, &tok) in ids[start..].iter().enumerate() {
        let pos = pad_len + offset;
        window[pos * v + tok as usize] = 1.0;
    }
    Tensor::new(vec![1, context_size * v], window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenizerMode;

    #[test]
    fn pretraining_tensors_have_correct_shape() {
        let vocab = Vocabulary::from_corpus(TokenizerMode::Char, &["abcabc"]);
        let v = vocab.size();
        let (x, y) = build_pretraining_tensors("abcabc", &vocab, 2).unwrap();
        // 6 tokens, context 2 → 4 examples
        assert_eq!(x.shape, vec![4, 2 * v]);
        assert_eq!(y.shape, vec![4, v]);
        // First example targets 'c' (third character)
        let target_id = vocab.id_of("c") as usize;
        assert_eq!(y.data[target_id], 1.0);
    }

    #[test]
    fn finetuning_emits_examples() {
        let vocab = Vocabulary::from_corpus(TokenizerMode::Char, &["hi there"]);
        let v = vocab.size();
        let pairs = vec![Pair { input: "hi".into(), output: "there".into() }];
        let (x, _y) = build_finetuning_tensors(&pairs, &vocab, 4, false).unwrap();
        // Sequence: <bos> h i <eos> t h e r e <eos> = 10 tokens, 9 transitions
        assert_eq!(x.cols(), 4 * v);
        assert_eq!(x.rows(), 9);
    }

    #[test]
    fn encode_context_left_pads() {
        let vocab = Vocabulary::from_corpus(TokenizerMode::Char, &["ab"]);
        let v = vocab.size();
        let ids = vocab.encode("a");
        let ctx = encode_context(&ids, &vocab, 3);
        assert_eq!(ctx.shape, vec![1, 3 * v]);
        // First two positions should be PAD, third should be 'a'
        assert_eq!(ctx.data[0 * v + PAD_ID as usize], 1.0);
        assert_eq!(ctx.data[1 * v + PAD_ID as usize], 1.0);
        assert_eq!(ctx.data[2 * v + vocab.id_of("a") as usize], 1.0);
    }
}
