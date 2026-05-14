use axum::{
    routing::{get, post, delete},
    Router,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

mod handlers;
mod models;
mod ws;

use models::*;

#[derive(Clone)]
struct AppState {
    networks: Arc<RwLock<HashMap<String, Network>>>,
    datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    trainers: Arc<RwLock<HashMap<String, Arc<RwLock<TrainingState>>>>>,
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

    let state = AppState::new();

    let app = Router::new()
        // Networks
        .route("/api/networks", post(handlers::create_network))
        .route("/api/networks", get(handlers::list_networks))
        .route("/api/networks/:id", get(handlers::get_network))
        .route("/api/networks/:id", delete(handlers::delete_network))
        // Datasets
        .route("/api/datasets", post(handlers::create_dataset))
        .route("/api/datasets", get(handlers::list_datasets))
        // Training
        .route("/api/train", post(handlers::start_training))
        .route("/api/train/:training_id", get(handlers::get_training_status))
        .route("/ws/train/:training_id", get(handlers::training_websocket))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3001")
        .await
        .expect("Failed to bind to 127.0.0.1:3001");

    tracing::info!("NeuralCabin backend server running on http://127.0.0.1:3001");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
