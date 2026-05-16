//! Disk persistence for the entire app state.
//!
//! Everything lives in `<app_data_dir>/state.json` (one file, atomically
//! replaced via a tempfile + rename). Writes happen synchronously after every
//! mutating command — the state is small enough that this is fine, and it
//! removes any window where data could be lost on crash.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use neuralcabin_engine::nn::Model;
use neuralcabin_engine::tokenizer::Vocabulary;

use crate::models::{Corpus, Network, TrainingRun, VocabularyInfo};
use crate::{AppState, VocabEntry};

pub const STATE_FILENAME: &str = "state.json";
pub const FORMAT_VERSION: u32 = 1;

/// Top-level on-disk snapshot of the full workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedState {
    pub format_version: u32,
    #[serde(default)]
    pub networks: HashMap<String, Network>,
    #[serde(default)]
    pub corpora: HashMap<String, Corpus>,
    #[serde(default)]
    pub vocabs: HashMap<String, PersistedVocab>,
    #[serde(default)]
    pub models: HashMap<String, Model>,
    /// Per-network training run history, keyed by network id.
    #[serde(default)]
    pub training_history: HashMap<String, Vec<TrainingRun>>,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            networks: HashMap::new(),
            corpora: HashMap::new(),
            vocabs: HashMap::new(),
            models: HashMap::new(),
            training_history: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVocab {
    pub info: VocabularyInfo,
    pub vocab: Vocabulary,
}

/// Build a `PersistedState` snapshot from the live `AppState`.
pub async fn snapshot(state: &AppState) -> PersistedState {
    let networks = state.networks.read().await.clone();
    let corpora = state.corpora.read().await.clone();

    let mut vocabs: HashMap<String, PersistedVocab> = HashMap::new();
    {
        let live = state.vocabs.read().await;
        for (k, e) in live.iter() {
            let inner = e.vocab.read().await.clone();
            vocabs.insert(k.clone(), PersistedVocab {
                info: e.info.clone(),
                vocab: inner,
            });
        }
    }

    let mut models: HashMap<String, Model> = HashMap::new();
    {
        let live = state.models.read().await;
        for (k, m) in live.iter() {
            let model = m.read().await.clone();
            models.insert(k.clone(), model);
        }
    }

    let training_history = state.training_history.read().await.clone();

    PersistedState { format_version: FORMAT_VERSION, networks, corpora, vocabs, models, training_history }
}

/// Persist the full state to `<data_dir>/state.json`. The write is atomic:
/// we write to `state.json.tmp` first and then rename over the target so that
/// a crash mid-write cannot leave the file half-written.
pub async fn save_to_dir(state: &AppState, data_dir: &Path) -> Result<(), String> {
    let persisted = snapshot(state).await;
    save_persisted(&persisted, data_dir).await
}

pub async fn save_persisted(persisted: &PersistedState, data_dir: &Path) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(persisted)
        .map_err(|e| format!("serialize state: {e}"))?;
    tokio::fs::create_dir_all(data_dir).await
        .map_err(|e| format!("create data dir {}: {e}", data_dir.display()))?;
    let final_path = data_dir.join(STATE_FILENAME);
    let tmp_path = data_dir.join(format!("{}.tmp", STATE_FILENAME));
    tokio::fs::write(&tmp_path, &json).await
        .map_err(|e| format!("write state tmp: {e}"))?;
    tokio::fs::rename(&tmp_path, &final_path).await
        .map_err(|e| format!("rename state file: {e}"))?;
    Ok(())
}

/// Load `state.json` from `data_dir`. Returns `Ok(None)` if the file doesn't
/// exist yet (fresh install). Returns an error if the file exists but is
/// corrupt — we DON'T silently wipe a corrupt state file, since that risks
/// destroying the user's work.
pub async fn load_from_dir(data_dir: &Path) -> Result<Option<PersistedState>, String> {
    let path = data_dir.join(STATE_FILENAME);
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    let persisted: PersistedState = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    if persisted.format_version > FORMAT_VERSION {
        return Err(format!(
            "state.json format_version {} is newer than this build supports ({})",
            persisted.format_version, FORMAT_VERSION
        ));
    }
    Ok(Some(persisted))
}

/// Populate an empty `AppState` from a loaded snapshot. Safe to call before
/// the app is registered as a Tauri-managed state.
pub async fn apply(state: &AppState, persisted: PersistedState) {
    *state.networks.write().await = persisted.networks;
    *state.corpora.write().await = persisted.corpora;
    let mut vocabs = state.vocabs.write().await;
    vocabs.clear();
    for (k, mut pv) in persisted.vocabs {
        // `Vocabulary` skips its lookup index during (de)serialization to keep
        // state.json small; rebuild it now so `encode` works on the very next
        // command after load.
        pv.vocab.rebuild_index();
        vocabs.insert(k, VocabEntry {
            vocab: Arc::new(RwLock::new(pv.vocab)),
            info: pv.info,
        });
    }
    let mut models = state.models.write().await;
    models.clear();
    for (k, m) in persisted.models {
        models.insert(k, Arc::new(RwLock::new(m)));
    }
    *state.training_history.write().await = persisted.training_history;
}

/// Return the configured data directory, or None if persistence is disabled
/// (e.g. during unit tests that don't go through the Tauri setup hook).
pub async fn data_dir(state: &AppState) -> Option<PathBuf> {
    state.data_dir.read().await.clone()
}

/// Best-effort save. Used by background tasks (e.g. training-loop) that have
/// nowhere to return an error. Failure is logged to stderr but does not panic.
pub async fn save_best_effort(state: &AppState) {
    let Some(dir) = data_dir(state).await else { return; };
    if let Err(e) = save_to_dir(state, &dir).await {
        eprintln!("[neuralcabin] failed to persist state: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AppState;
    use crate::models::{LayerDef, kinds};
    use chrono::Utc;
    use neuralcabin_engine::nn::{LayerSpec, Model};
    use neuralcabin_engine::activations::Activation as EngineActivation;
    use neuralcabin_engine::tokenizer::{TokenizerMode, Vocabulary, VocabularyOptions};

    fn sample_network(id: &str) -> Network {
        Network {
            id: id.to_string(),
            name: format!("net-{id}"),
            kind: kinds::FEEDFORWARD.to_string(),
            seed: 42,
            created_at: Utc::now(),
            trained: false,
            input_dim: 2,
            output_dim: 1,
            layers: vec![
                LayerDef::Linear { in_dim: 2, out_dim: 4 },
                LayerDef::Activation { activation: "relu".to_string() },
                LayerDef::Linear { in_dim: 4, out_dim: 1 },
            ],
            parameter_count: 17,
            hidden_layers: None,
            context_size: None,
        }
    }

    fn sample_model() -> Model {
        let specs = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 4 },
            LayerSpec::Activation(EngineActivation::ReLU),
            LayerSpec::Linear { in_dim: 4, out_dim: 1 },
        ];
        Model::from_specs(2, &specs, 7)
    }

    fn sample_vocab() -> Vocabulary {
        Vocabulary::build(
            TokenizerMode::Char,
            &["hello world", "the quick brown fox"],
            &VocabularyOptions::default(),
        )
    }

    #[tokio::test]
    async fn snapshot_and_apply_roundtrip() {
        let original = AppState::new();
        original.networks.write().await.insert("a".into(), sample_network("a"));
        original.networks.write().await.insert("b".into(), sample_network("b"));
        original.models.write().await.insert("a".into(), Arc::new(RwLock::new(sample_model())));
        original.corpora.write().await.insert("a".into(), Corpus {
            network_id: "a".into(),
            kind: kinds::FEEDFORWARD.into(),
            updated_at: Utc::now(),
            feedforward: None,
            text: Some("hello".into()),
            pairs: None,
            stage: Some("pretrain".into()),
        });
        let vocab = sample_vocab();
        original.vocabs.write().await.insert("a".into(), VocabEntry {
            vocab: Arc::new(RwLock::new(vocab.clone())),
            info: VocabularyInfo {
                mode: "char".into(),
                tokens: vocab.tokens.clone(),
                options: crate::models::VocabularyOptions::default(),
                updated_at: Utc::now(),
            },
        });

        let snap = snapshot(&original).await;
        let restored = AppState::new();
        apply(&restored, snap).await;

        assert_eq!(restored.networks.read().await.len(), 2);
        assert!(restored.networks.read().await.contains_key("a"));
        assert!(restored.networks.read().await.contains_key("b"));
        assert_eq!(restored.corpora.read().await.len(), 1);
        assert_eq!(restored.vocabs.read().await.len(), 1);
        assert_eq!(restored.models.read().await.len(), 1);

        let restored_model = restored.models.read().await.get("a").cloned().unwrap();
        assert_eq!(restored_model.read().await.input_dim, 2);
    }

    #[tokio::test]
    async fn save_and_load_from_disk() {
        let dir = tempdir();
        let original = AppState::new();
        original.networks.write().await.insert("net-1".into(), sample_network("net-1"));
        original.models.write().await.insert("net-1".into(), Arc::new(RwLock::new(sample_model())));

        save_to_dir(&original, &dir).await.expect("save");
        assert!(dir.join(STATE_FILENAME).exists());

        let loaded = load_from_dir(&dir).await.expect("load").expect("file exists");
        assert_eq!(loaded.format_version, FORMAT_VERSION);
        assert_eq!(loaded.networks.len(), 1);
        assert_eq!(loaded.models.len(), 1);

        let restored = AppState::new();
        apply(&restored, loaded).await;
        assert_eq!(restored.networks.read().await.get("net-1").unwrap().name, "net-net-1");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_returns_none_when_missing() {
        let dir = tempdir();
        let loaded = load_from_dir(&dir).await.expect("ok");
        assert!(loaded.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_errors_on_corrupt_file() {
        let dir = tempdir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(STATE_FILENAME), b"not json {").unwrap();
        let result = load_from_dir(&dir).await;
        assert!(result.is_err(), "corrupt file should produce an error, not silently drop the state");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("neuralcabin-test-{}", uuid::Uuid::new_v4()));
        p
    }
}
