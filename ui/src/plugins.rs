//! Plugin registry.
//!
//! A plugin is a third-party-defined "network type". A real ZIP archive can be
//! supplied — we look for an embedded `manifest.json` (uncompressed entries
//! only). Folders containing a `manifest.json` and bare `.json` manifest files
//! are also accepted, which keeps the registration UX usable regardless of how
//! the plugin author packaged the artefact.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub description: String,
    /// Network types this plugin contributes (these become selectable in the
    /// "Create Network" panel).
    #[serde(default)]
    pub network_types: Vec<String>,
    /// Whether this plugin owns the Vocab tab when one of its network types is
    /// active. If true the Vocab tab is enabled and labelled as plugin-managed.
    #[serde(default)]
    pub manages_vocab: bool,
    /// Whether the plugin owns the Inference tab. When false, plugin networks
    /// fall back to the Simplex inference UI.
    #[serde(default)]
    pub manages_inference: bool,
}

#[derive(Clone, Debug)]
pub struct PluginEntry {
    pub manifest: PluginManifest,
    pub source_path: PathBuf,
    /// User-tweakable settings, kept as a JSON string the plugin can later parse.
    pub settings_json: String,
}

#[derive(Default)]
pub struct PluginRegistry {
    pub plugins: Vec<PluginEntry>,
    pub message: Option<String>,
    pub upload_path: String,
    pub selected: Option<usize>,
}

impl PluginRegistry {
    #[allow(dead_code)]
    pub fn names(&self) -> Vec<String> {
        self.plugins.iter().map(|p| p.manifest.name.clone()).collect()
    }

    pub fn find_by_id(&self, id: &str) -> Option<&PluginEntry> {
        self.plugins.iter().find(|p| p.manifest.id == id)
    }

    pub fn all_network_types(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for p in &self.plugins {
            for ty in &p.manifest.network_types {
                out.push((p.manifest.id.clone(), ty.clone()));
            }
        }
        out
    }

    pub fn install(&mut self, path: &str) {
        match try_load(path) {
            Ok((manifest, source)) => {
                if self.plugins.iter().any(|p| p.manifest.id == manifest.id) {
                    self.message = Some(format!("Plugin id '{}' already installed.", manifest.id));
                    return;
                }
                let id = manifest.id.clone();
                self.plugins.push(PluginEntry {
                    manifest,
                    source_path: source,
                    settings_json: "{}".into(),
                });
                self.message = Some(format!("Installed plugin '{id}'."));
            }
            Err(e) => self.message = Some(format!("Install failed: {e}")),
        }
    }

    pub fn remove(&mut self, idx: usize) {
        if idx < self.plugins.len() {
            let id = self.plugins.remove(idx).manifest.id;
            self.message = Some(format!("Removed plugin '{id}'."));
            if self.selected == Some(idx) { self.selected = None; }
        }
    }
}

fn try_load(path: &str) -> Result<(PluginManifest, PathBuf), String> {
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("path '{path}' does not exist"));
    }
    if p.is_dir() {
        let manifest_path = p.join("manifest.json");
        let raw = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("read manifest.json: {e}"))?;
        return Ok((parse(&raw)?, p.to_path_buf()));
    }
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "json" => {
            let raw = fs::read_to_string(p).map_err(|e| e.to_string())?;
            Ok((parse(&raw)?, p.to_path_buf()))
        }
        "zip" => {
            let raw = read_manifest_from_zip(p)?;
            Ok((parse(&raw)?, p.to_path_buf()))
        }
        other => Err(format!("unsupported plugin source '.{other}'")),
    }
}

fn parse(raw: &str) -> Result<PluginManifest, String> {
    let m: PluginManifest = serde_json::from_str(raw).map_err(|e| format!("manifest JSON: {e}"))?;
    if m.id.trim().is_empty() { return Err("manifest.id is empty".into()); }
    if m.name.trim().is_empty() { return Err("manifest.name is empty".into()); }
    Ok(m)
}

/// Minimal ZIP reader: walks Local File Headers, returning the bytes of an
/// embedded `manifest.json`. Only the "stored" (no compression) method is
/// supported — plugins that want to ship as a `.zip` should store their
/// manifest uncompressed (the rest of the archive can be compressed). Returns
/// the manifest as a UTF-8 string.
fn read_manifest_from_zip(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("read zip: {e}"))?;
    let mut i = 0usize;
    while i + 30 <= bytes.len() {
        let sig = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        if sig != 0x0403_4b50 { break; }
        let method = u16::from_le_bytes(bytes[i + 8..i + 10].try_into().unwrap());
        let comp_size = u32::from_le_bytes(bytes[i + 18..i + 22].try_into().unwrap()) as usize;
        let uncomp_size = u32::from_le_bytes(bytes[i + 22..i + 26].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes(bytes[i + 26..i + 28].try_into().unwrap()) as usize;
        let extra_len = u16::from_le_bytes(bytes[i + 28..i + 30].try_into().unwrap()) as usize;
        let header_end = i + 30 + name_len + extra_len;
        if header_end + comp_size > bytes.len() { break; }
        let name = std::str::from_utf8(&bytes[i + 30..i + 30 + name_len]).unwrap_or("");
        let basename = name.rsplit('/').next().unwrap_or(name);
        if basename == "manifest.json" {
            if method != 0 {
                return Err(format!(
                    "manifest.json in '{}' is compressed (method {method}); store it uncompressed.",
                    path.display()
                ));
            }
            let raw = &bytes[header_end..header_end + comp_size];
            let s = std::str::from_utf8(raw).map_err(|e| e.to_string())?;
            let _ = uncomp_size;
            return Ok(s.to_string());
        }
        i = header_end + comp_size;
    }
    Err(format!("no manifest.json found in '{}'.", path.display()))
}
