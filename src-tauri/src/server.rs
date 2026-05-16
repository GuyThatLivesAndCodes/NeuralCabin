//! Embedded HTTP server for remote network operations.
//!
//! Users can start one or more servers from the Server tab. Each listens on a
//! configurable port and exposes a REST + JSON API that mirrors the most
//! useful Tauri commands: list networks, fetch model weights, upload weights,
//! kick off training, poll training status, export, and run inference.
//!
//! Servers are first-class objects in `AppState.servers`, persisted to
//! `state.json` alongside everything else. Starting / stopping a server only
//! changes its in-memory `JoinHandle`; the configuration survives restarts.
//!
//! Auth: each server can optionally require a bearer token. Clients send it
//! via the `Authorization: Bearer <token>` header. With no token configured,
//! the server is open to anyone who can reach the port — appropriate for
//! LAN-only setups, **not** for the public internet.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{Path as AxumPath, Query, State as AxumState},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock};
use tokio::task::JoinHandle;
use tauri::{AppHandle, Manager};
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::AppState;
use crate::export;
use neuralcabin_engine::nn::Model;

// ─── Public types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerPermissions {
    #[serde(default = "default_true")]
    pub allow_list: bool,
    #[serde(default = "default_true")]
    pub allow_inference: bool,
    #[serde(default = "default_true")]
    pub allow_export: bool,
    #[serde(default)]
    pub allow_upload: bool,
    #[serde(default)]
    pub allow_train: bool,
    #[serde(default)]
    pub allow_create: bool,
    #[serde(default)]
    pub allow_delete: bool,
}
fn default_true() -> bool { true }

impl Default for ServerPermissions {
    fn default() -> Self {
        Self {
            allow_list: true, allow_inference: true, allow_export: true,
            allow_upload: false, allow_train: false,
            allow_create: false, allow_delete: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub id: String,
    pub name: String,
    pub port: u16,
    /// Bind to localhost only (true) or all interfaces (false).
    #[serde(default = "default_true")]
    pub localhost_only: bool,
    /// Optional bearer token. Empty string = no auth.
    #[serde(default)]
    pub auth_token: String,
    #[serde(default)]
    pub permissions: ServerPermissions,
    /// Restart this server on app startup. The UI manages the flag; the
    /// background task at startup reads it to decide which servers to bring
    /// up.
    #[serde(default)]
    pub auto_start: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub id: String,
    pub running: bool,
    /// Number of requests served since this server was last started.
    pub request_count: u64,
    /// Last error message, if a previous start failed.
    pub last_error: Option<String>,
}

pub struct ServerRuntime {
    pub config: ServerConfig,
    pub handle: Option<JoinHandle<()>>,
    pub shutdown: Arc<Notify>,
    pub request_count: Arc<std::sync::atomic::AtomicU64>,
    pub running: Arc<AtomicBool>,
    pub last_error: Arc<RwLock<Option<String>>>,
}

impl ServerRuntime {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            handle: None,
            shutdown: Arc::new(Notify::new()),
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(false)),
            last_error: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn status(&self) -> ServerStatus {
        ServerStatus {
            id: self.config.id.clone(),
            running: self.running.load(Ordering::Relaxed),
            request_count: self.request_count.load(Ordering::Relaxed),
            last_error: self.last_error.read().await.clone(),
        }
    }
}

// ─── Lifecycle ─────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HttpCtx {
    app: AppHandle,
    config: Arc<RwLock<ServerConfig>>,
    request_count: Arc<std::sync::atomic::AtomicU64>,
}

impl HttpCtx {
    fn state(&self) -> tauri::State<'_, AppState> { self.app.state::<AppState>() }
}

pub async fn start(
    app: AppHandle,
    runtime: &mut ServerRuntime,
) -> Result<(), String> {
    if runtime.running.load(Ordering::Relaxed) {
        return Err("server already running".into());
    }
    let bind_ip = if runtime.config.localhost_only { "127.0.0.1" } else { "0.0.0.0" };
    let addr: SocketAddr = format!("{bind_ip}:{}", runtime.config.port)
        .parse()
        .map_err(|e| format!("invalid bind address: {e}"))?;

    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| format!("bind {addr}: {e}"))?;

    let ctx = HttpCtx {
        app: app.clone(),
        config: Arc::new(RwLock::new(runtime.config.clone())),
        request_count: runtime.request_count.clone(),
    };

    let router = build_router(ctx);
    let shutdown = runtime.shutdown.clone();
    let running = runtime.running.clone();
    let last_error = runtime.last_error.clone();

    runtime.request_count.store(0, Ordering::Relaxed);
    running.store(true, Ordering::Relaxed);
    *last_error.write().await = None;

    let handle = tokio::spawn(async move {
        let serve = axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown.notified().await; });
        if let Err(e) = serve.await {
            *last_error.write().await = Some(format!("server error: {e}"));
        }
        running.store(false, Ordering::Relaxed);
    });
    runtime.handle = Some(handle);
    Ok(())
}

pub async fn stop(runtime: &mut ServerRuntime) -> Result<(), String> {
    if !runtime.running.load(Ordering::Relaxed) {
        return Ok(());
    }
    runtime.shutdown.notify_waiters();
    if let Some(h) = runtime.handle.take() {
        // Don't await indefinitely; the shutdown future completes promptly.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), h).await;
    }
    runtime.running.store(false, Ordering::Relaxed);
    Ok(())
}

// ─── Router ────────────────────────────────────────────────────────────────

fn build_router(ctx: HttpCtx) -> Router {
    Router::new()
        .route("/",                          get(root_index))
        .route("/api/info",                  get(api_info))
        .route("/api/networks",              get(api_list_networks).post(api_create_network))
        .route("/api/networks/:id",          get(api_get_network).delete(api_delete_network))
        .route("/api/networks/:id/weights",  get(api_download_weights).post(api_upload_weights))
        .route("/api/networks/:id/export",   get(api_export))
        .route("/api/networks/:id/infer",    post(api_infer))
        .route("/api/networks/:id/train",    post(api_train))
        .route("/api/training/:tid",         get(api_training_status))
        .layer(CorsLayer::very_permissive())
        .with_state(ctx)
}

// ─── Auth + middleware-ish helpers ─────────────────────────────────────────

async fn check_auth(ctx: &HttpCtx, headers: &HeaderMap) -> Result<(), Response> {
    let cfg = ctx.config.read().await;
    if cfg.auth_token.is_empty() {
        return Ok(());
    }
    let provided = headers.get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(tok) if tok == cfg.auth_token => Ok(()),
        _ => Err((StatusCode::UNAUTHORIZED, "missing or invalid Bearer token").into_response()),
    }
}

async fn require_perm(
    ctx: &HttpCtx,
    pick: impl Fn(&ServerPermissions) -> bool,
    name: &str,
) -> Result<(), Response> {
    let cfg = ctx.config.read().await;
    if pick(&cfg.permissions) { Ok(()) }
    else { Err((StatusCode::FORBIDDEN, format!("permission denied: {name} disabled on this server")).into_response()) }
}

fn count(ctx: &HttpCtx) {
    ctx.request_count.fetch_add(1, Ordering::Relaxed);
}

fn json_err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}

// ─── Handlers ──────────────────────────────────────────────────────────────

async fn root_index(AxumState(ctx): AxumState<HttpCtx>) -> impl IntoResponse {
    count(&ctx);
    let cfg = ctx.config.read().await.clone();
    let html = format!(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NeuralCabin · {name}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; color: #222; }}
code {{ background: #f4f4f8; padding: 1px 5px; border-radius: 4px; }}
pre  {{ background: #f4f4f8; padding: 12px; border-radius: 6px; overflow-x: auto; }}
.tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px; background: #e9eef7; margin-right: 4px; font-size: 0.85em; }}
.deny {{ background: #f7eaea; color: #883; }}
</style></head><body>
<h1>NeuralCabin Server</h1>
<p><strong>{name}</strong> on port <code>{port}</code>. Auth: {auth}.</p>
<h2>Permissions</h2>
<p>
<span class="tag {list_cls}">list</span>
<span class="tag {infer_cls}">infer</span>
<span class="tag {export_cls}">export</span>
<span class="tag {upload_cls}">upload</span>
<span class="tag {train_cls}">train</span>
<span class="tag {create_cls}">create</span>
<span class="tag {delete_cls}">delete</span>
</p>

<h2>Endpoints</h2>
<ul>
  <li><code>GET  /api/info</code> — server config &amp; permissions</li>
  <li><code>GET  /api/networks</code> — list networks</li>
  <li><code>GET  /api/networks/{{id}}</code> — network metadata</li>
  <li><code>POST /api/networks</code> — create a network <span class="tag {create_cls}">create</span></li>
  <li><code>DELETE /api/networks/{{id}}</code> — delete a network <span class="tag {delete_cls}">delete</span></li>
  <li><code>GET  /api/networks/{{id}}/weights</code> — download model JSON <span class="tag {export_cls}">export</span></li>
  <li><code>POST /api/networks/{{id}}/weights</code> — upload model JSON <span class="tag {upload_cls}">upload</span></li>
  <li><code>GET  /api/networks/{{id}}/export?format=pytorch|onnx|gguf</code> <span class="tag {export_cls}">export</span></li>
  <li><code>POST /api/networks/{{id}}/infer</code> — run inference <span class="tag {infer_cls}">infer</span></li>
  <li><code>POST /api/networks/{{id}}/train</code> — start training <span class="tag {train_cls}">train</span></li>
  <li><code>GET  /api/training/{{tid}}</code> — poll training status</li>
</ul>

<h2>Example: remote training workflow</h2>
<p>A remote user can upload their own weights to your computer, train them
here, and download the updated weights:</p>
<pre>
# 1. Upload weights (a JSON dump of the NeuralCabin model format)
curl -X POST {auth_hdr}http://HOST:{port}/api/networks/NET_ID/weights \
     -H 'Content-Type: application/json' --data-binary @model.json

# 2. Kick training
curl -X POST {auth_hdr}http://HOST:{port}/api/networks/NET_ID/train \
     -H 'Content-Type: application/json' \
     -d '{{"epochs":50,"batch_size":16,"loss":"mse","seed":1,"optimizer":{{"kind":"adam","lr":0.01}}}}'

# 3. Poll
curl {auth_hdr}http://HOST:{port}/api/training/TRAINING_ID

# 4. Download trained weights
curl {auth_hdr}http://HOST:{port}/api/networks/NET_ID/weights -o trained.json

# 5. (or) Export for another framework
curl {auth_hdr}'http://HOST:{port}/api/networks/NET_ID/export?format=onnx' -o model.onnx
</pre>

<p style="margin-top:2rem;color:#777;font-size:0.9em">
The owner of this machine (you) controls start/stop, the port, the auth
token, and which operations remote users are allowed to perform from the
Server tab in NeuralCabin.
</p>
</body></html>"#,
        name = html_escape(&cfg.name),
        port = cfg.port,
        auth = if cfg.auth_token.is_empty() { "open (no token required)" } else { "Bearer token required" },
        auth_hdr = if cfg.auth_token.is_empty() { String::new() } else { format!("-H 'Authorization: Bearer {}' ", cfg.auth_token) },
        list_cls   = perm_cls(cfg.permissions.allow_list),
        infer_cls  = perm_cls(cfg.permissions.allow_inference),
        export_cls = perm_cls(cfg.permissions.allow_export),
        upload_cls = perm_cls(cfg.permissions.allow_upload),
        train_cls  = perm_cls(cfg.permissions.allow_train),
        create_cls = perm_cls(cfg.permissions.allow_create),
        delete_cls = perm_cls(cfg.permissions.allow_delete),
    );

    Response::builder()
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(axum::body::Body::from(html))
        .unwrap()
}

fn perm_cls(allowed: bool) -> &'static str {
    if allowed { "" } else { "deny" }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

async fn api_info(
    AxumState(ctx): AxumState<HttpCtx>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    let cfg = ctx.config.read().await.clone();
    Json(serde_json::json!({
        "name": cfg.name,
        "port": cfg.port,
        "permissions": cfg.permissions,
        "auth_required": !cfg.auth_token.is_empty(),
    })).into_response()
}

async fn api_list_networks(
    AxumState(ctx): AxumState<HttpCtx>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_list, "list").await { return r; }

    let nets: Vec<_> = ctx.state().networks.read().await.values().cloned().collect();
    Json(nets).into_response()
}

async fn api_get_network(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_list, "list").await { return r; }
    match ctx.state().networks.read().await.get(&id).cloned() {
        Some(n) => Json(n).into_response(),
        None    => json_err(StatusCode::NOT_FOUND, "network not found"),
    }
}

async fn api_create_network(
    AxumState(ctx): AxumState<HttpCtx>,
    headers: HeaderMap,
    Json(req): Json<crate::models::CreateNetworkRequest>,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_create, "create").await { return r; }

    match crate::create_network_internal(&ctx.app, req).await {
        Ok(n) => Json(n).into_response(),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn api_delete_network(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_delete, "delete").await { return r; }

    match crate::delete_network_internal(&ctx.app, &id).await {
        Ok(removed) => Json(serde_json::json!({ "removed": removed })).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn api_download_weights(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_export, "export").await { return r; }

    let net = match ctx.state().networks.read().await.get(&id).cloned() {
        Some(n) => n,
        None => return json_err(StatusCode::NOT_FOUND, "network not found"),
    };
    let model = match crate::get_or_load_model_arc(&ctx.app, &net).await {
        Ok(m) => m, Err(e) => return json_err(StatusCode::BAD_REQUEST, e),
    };
    let snapshot = model.read().await.clone();
    let bytes = match serde_json::to_vec(&snapshot) {
        Ok(b) => b, Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("serialize: {e}")),
    };
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}-weights.json\"", sanitize_filename(&net.name)),
        )
        .body(bytes.into())
        .unwrap()
}

async fn api_upload_weights(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_upload, "upload").await { return r; }

    let net = match ctx.state().networks.read().await.get(&id).cloned() {
        Some(n) => n,
        None => return json_err(StatusCode::NOT_FOUND, "network not found"),
    };

    let uploaded: Model = match serde_json::from_slice(&body) {
        Ok(m) => m, Err(e) => return json_err(StatusCode::BAD_REQUEST, format!("invalid model JSON: {e}")),
    };
    if uploaded.input_dim != net.input_dim {
        return json_err(StatusCode::BAD_REQUEST, format!(
            "uploaded model input_dim {} != network input_dim {}",
            uploaded.input_dim, net.input_dim
        ));
    }
    if uploaded.output_dim() != net.output_dim {
        return json_err(StatusCode::BAD_REQUEST, format!(
            "uploaded model output_dim {} != network output_dim {}",
            uploaded.output_dim(), net.output_dim
        ));
    }
    crate::install_model_for_network(&ctx.app, &net, uploaded).await;
    Json(serde_json::json!({ "ok": true })).into_response()
}

#[derive(Deserialize)]
struct ExportQuery { format: String }

async fn api_export(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<ExportQuery>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_export, "export").await { return r; }

    let net = match ctx.state().networks.read().await.get(&id).cloned() {
        Some(n) => n,
        None => return json_err(StatusCode::NOT_FOUND, "network not found"),
    };
    let model = match crate::get_or_load_model_arc(&ctx.app, &net).await {
        Ok(m) => m, Err(e) => return json_err(StatusCode::BAD_REQUEST, e),
    };
    let snapshot = model.read().await.clone();
    match export::export_model(&q.format, &net, &snapshot) {
        Ok(bytes) => {
            let ext = export::extension_for(&q.format);
            Response::builder()
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{}.{}\"", sanitize_filename(&net.name), ext),
                )
                .body(bytes.into())
                .unwrap()
        }
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn api_infer(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
    Json(mut req): Json<crate::models::InferRequest>,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_inference, "inference").await { return r; }

    req.network_id = id;
    match crate::infer_sync(&ctx.app, req).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn api_train(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
    Json(cfg): Json<crate::models::TrainingConfig>,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }
    if let Err(r) = require_perm(&ctx, |p| p.allow_train, "train").await { return r; }

    let req = crate::models::StartTrainingRequest { network_id: id, config: cfg };
    match crate::start_training_internal(&ctx.app, req).await {
        Ok(r) => Json(r).into_response(),
        Err(e) => json_err(StatusCode::BAD_REQUEST, e),
    }
}

async fn api_training_status(
    AxumState(ctx): AxumState<HttpCtx>,
    AxumPath(tid): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    count(&ctx);
    if let Err(r) = check_auth(&ctx, &headers).await { return r; }

    let state = ctx.state();
    let trainers = state.trainers.read().await;
    let Some(h) = trainers.get(&tid).cloned() else {
        return json_err(StatusCode::NOT_FOUND, "training session not found");
    };
    drop(trainers);
    let s = h.state.read().await;
    let status_str = if s.error.is_some() { "error" }
                     else if s.cancelled { "cancelled" }
                     else if s.stopped { "completed" }
                     else { "running" };
    Json(serde_json::json!({
        "training_id": tid,
        "status": status_str,
        "epoch": s.epoch,
        "total_epochs": s.total_epochs,
        "last_loss": s.last_loss,
        "loss_history": s.loss_history,
        "elapsed_secs": s.elapsed_secs,
        "error": s.error,
    })).into_response()
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

// ─── Server registry helpers ───────────────────────────────────────────────

pub fn new_config(name: String, port: u16) -> ServerConfig {
    ServerConfig {
        id: Uuid::new_v4().to_string(),
        name, port,
        localhost_only: true,
        auth_token: String::new(),
        permissions: ServerPermissions::default(),
        auto_start: false,
        created_at: Utc::now(),
    }
}

#[allow(dead_code)]
pub type ServerMap = HashMap<String, ServerRuntime>;
