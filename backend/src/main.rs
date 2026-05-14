use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post, delete},
    Router,
};
use rust_embed::RustEmbed;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

mod handlers;
mod models;
mod ws;

use models::*;

/// Embeds the compiled React app from `frontend/dist` at compile time.
/// Run `cd frontend && npm run build` before `cargo build`.
#[derive(RustEmbed)]
#[folder = "../frontend/dist"]
struct FrontendAssets;

#[derive(Clone)]
pub struct AppState {
    pub networks: Arc<RwLock<HashMap<String, Network>>>,
    pub datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    pub trainers: Arc<RwLock<HashMap<String, Arc<RwLock<TrainingState>>>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            networks: Arc::new(RwLock::new(HashMap::new())),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            trainers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    // Support headless XOR demo mode
    if args.iter().any(|a| a == "--xor-demo") {
        headless_xor_demo();
        return;
    }

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(3001);

    let state = AppState::new();

    let app = Router::new()
        // REST API
        .route("/api/networks",          post(handlers::create_network))
        .route("/api/networks",          get(handlers::list_networks))
        .route("/api/networks/:id",      get(handlers::get_network))
        .route("/api/networks/:id",      delete(handlers::delete_network))
        .route("/api/datasets",          post(handlers::create_dataset))
        .route("/api/datasets",          get(handlers::list_datasets))
        .route("/api/train",             post(handlers::start_training))
        .route("/api/train/:training_id", get(handlers::get_training_status))
        // WebSocket
        .route("/ws/train/:training_id", get(handlers::training_websocket))
        // Embedded frontend — serves React app for every other path
        .fallback(serve_frontend)
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));

    let url = format!("http://localhost:{port}");
    println!("🧠 NeuralCabin is running!");
    println!("   Open your browser: {url}");
    println!("   Press Ctrl+C to stop.");

    // Try to open browser in background (don't block server start)
    let url_clone = url.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let _ = open_browser(&url_clone);
    });

    axum::serve(listener, app).await.expect("Server error");
}

/// Headless XOR training demo (for CI testing)
fn headless_xor_demo() {
    use neuralcabin_engine::nn::{LayerSpec, Model};
    use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
    use neuralcabin_engine::tensor::Tensor;
    use neuralcabin_engine::{Activation, Loss};

    println!("Training XOR MLP (headless)...");
    let specs = vec![
        LayerSpec::Linear { in_dim: 2, out_dim: 8 },
        LayerSpec::Activation(Activation::Tanh),
        LayerSpec::Linear { in_dim: 8, out_dim: 1 },
        LayerSpec::Activation(Activation::Sigmoid),
    ];
    let mut model = Model::from_specs(2, &specs, 42);
    let mut opt = Optimizer::new(
        OptimizerKind::Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        &model.parameter_shapes(),
    );
    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);
    let mut last = f32::INFINITY;
    for epoch in 1..=2000 {
        last = model.train_step(&mut opt, Loss::MeanSquaredError, &x, &y);
        if epoch % 200 == 0 {
            println!("  epoch {epoch:>5}  loss = {last:.6}");
        }
    }
    println!("Final loss = {last:.6}");
    let pred = model.predict(&x);
    println!("Predictions:");
    for (i, p) in pred.data.iter().enumerate() {
        let xi = &x.data[i * 2..(i + 1) * 2];
        let ti = y.data[i];
        println!("  XOR({}, {}) -> {p:.4}  (target {ti})", xi[0], xi[1]);
    }
    if last > 0.05 {
        eprintln!("convergence check failed: final loss {last} > 0.05");
        std::process::exit(1);
    }
}

async fn serve_frontend(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');

    // Try the exact path first, then fall back to index.html for SPA routing
    let asset = FrontendAssets::get(path)
        .or_else(|| {
            // Files with extensions that aren't found → 404
            if path.contains('.') {
                None
            } else {
                FrontendAssets::get("index.html")
            }
        });

    match asset {
        Some(content) => {
            let mime = mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string();
            Response::builder()
                .header(header::CONTENT_TYPE, mime)
                .body(Body::from(content.data))
                .unwrap()
        }
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    std::process::Command::new("cmd")
        .args(["/c", "start", url])
        .spawn()?;

    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(url).spawn()?;

    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(url).spawn()?;

    Ok(())
}
