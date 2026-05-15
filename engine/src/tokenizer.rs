//! Tokenization utilities for next-token-prediction networks.
//!
//! Provides two real tokenization strategies:
//! - `Char`: one token per Unicode character (smallest vocab, longest sequences)
//! - `Word`: whitespace-split tokens with punctuation kept as separate tokens
//!
//! Both produce a [`Vocabulary`] with reserved special tokens at fixed indices:
//!   0 = `<pad>` (padding for batches/contexts shorter than required)
//!   1 = `<unk>` (anything not in the trained vocab)
//!   2 = `<bos>` (beginning of sequence)
//!   3 = `<eos>` (end of sequence)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const RESERVED: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerMode {
    Char,
    Word,
}

impl TokenizerMode {
    pub fn name(&self) -> &'static str {
        match self {
            TokenizerMode::Char => "Char",
            TokenizerMode::Word => "Word",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    pub mode: TokenizerMode,
    /// Index → token string. Indices 0..RESERVED are special tokens.
    pub tokens: Vec<String>,
    /// Token string → index lookup.
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl Vocabulary {
    pub fn new(mode: TokenizerMode) -> Self {
        let tokens = vec![
            "<pad>".to_string(),
            "<unk>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
        ];
        let mut index = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            index.insert(t.clone(), i as u32);
        }
        Self { mode, tokens, index }
    }

    /// Rebuild the lookup map after deserialisation.
    pub fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, t) in self.tokens.iter().enumerate() {
            self.index.insert(t.clone(), i as u32);
        }
    }

    pub fn size(&self) -> usize { self.tokens.len() }

    pub fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.index.get(&token) {
            return id;
        }
        let id = self.tokens.len() as u32;
        self.index.insert(token.clone(), id);
        self.tokens.push(token);
        id
    }

    pub fn id_of(&self, token: &str) -> u32 {
        self.index.get(token).copied().unwrap_or(UNK_ID)
    }

    pub fn token_of(&self, id: u32) -> &str {
        self.tokens
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("<unk>")
    }

    /// Build a vocabulary from a text corpus. Empty `texts` yields a vocab with
    /// only the reserved special tokens.
    pub fn from_corpus(mode: TokenizerMode, texts: &[&str]) -> Self {
        let mut vocab = Vocabulary::new(mode);
        for text in texts {
            for tok in tokenize(mode, text) {
                vocab.add_token(tok);
            }
        }
        vocab
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        tokenize(self.mode, text)
            .into_iter()
            .map(|t| self.id_of(&t))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for (i, &id) in ids.iter().enumerate() {
            let tok = self.token_of(id);
            // Special tokens are not rendered to user-facing text
            if (id as usize) < RESERVED { continue; }
            match self.mode {
                TokenizerMode::Char => out.push_str(tok),
                TokenizerMode::Word => {
                    if i > 0 && !is_punct_only(tok) && !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(tok);
                }
            }
        }
        out
    }
}

fn is_punct_only(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_punctuation())
}

/// Pure tokenization with no vocab — returns the surface tokens.
pub fn tokenize(mode: TokenizerMode, text: &str) -> Vec<String> {
    match mode {
        TokenizerMode::Char => text.chars().map(|c| c.to_string()).collect(),
        TokenizerMode::Word => word_tokenize(text),
    }
}

/// Whitespace-and-punctuation tokenizer. Keeps punctuation as separate tokens
/// so the model can learn sentence structure without exploding vocab size on
/// "word." vs "word".
fn word_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
        } else if c.is_ascii_punctuation() {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.push(c.to_string());
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() { out.push(current); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_tokenizer_roundtrip() {
        let vocab = Vocabulary::from_corpus(TokenizerMode::Char, &["hello"]);
        assert!(vocab.size() >= RESERVED + 4); // h, e, l, o
        let ids = vocab.encode("hello");
        assert_eq!(ids.len(), 5);
        assert_eq!(vocab.decode(&ids), "hello");
    }

    #[test]
    fn word_tokenizer_splits_punct() {
        let toks = word_tokenize("Hello, world!");
        assert_eq!(toks, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn unknown_words_become_unk() {
        let vocab = Vocabulary::from_corpus(TokenizerMode::Word, &["the cat sat"]);
        let ids = vocab.encode("the dog");
        assert_eq!(ids[0], vocab.id_of("the"));
        assert_eq!(ids[1], UNK_ID);
    }
}
