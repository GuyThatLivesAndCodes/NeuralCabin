use axum::{
    extract::{Path, State, ws::WebSocketUpgrade},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use neuralcabin_engine::nn::{LayerSpec, Model};
use neuralcabin_engine::optimizer::OptimizerKind;
use neuralcabin_engine::tensor::Tensor;
use neuralcabin_engine::{Activation, Loss};
use std::sync::Arc;
use uuid::Uuid;

use crate::models::*;
use crate::ws;
use crate::AppState;

pub async fn create_network(
    State(state): State<AppState>,
    Json(req): Json<CreateNetworkRequest>,
) -> (StatusCode, Json<Network>) {
    let id = Uuid::new_v4().to_string();
    let network = Network {
        id: id.clone(),
        name: req.name,
        kind: req.kind,
        layers: req.layers,
        seed: req.seed,
        created_at: Utc::now(),
    };

    state.networks.write().await.insert(id, network.clone());
    (StatusCode::CREATED, Json(network))
}

pub async fn list_networks(
    State(state): State<AppState>,
) -> Json<ListNetworksResponse> {
    let networks = state.networks.read().await;
    let networks_vec: Vec<_> = networks.values().cloned().collect();
    Json(ListNetworksResponse {
        networks: networks_vec,
    })
}

pub async fn get_network(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Network>, (StatusCode, Json<ErrorResponse>)> {
    let networks = state.networks.read().await;

    networks.get(&id)
        .cloned()
        .map(Json)
        .ok_or((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Network not found".to_string(),
            }),
        ))
}

pub async fn delete_network(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> StatusCode {
    let mut networks = state.networks.write().await;

    if networks.remove(&id).is_some() {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

pub async fn create_dataset(
    State(state): State<AppState>,
    Json(req): Json<CreateDatasetRequest>,
) -> (StatusCode, Json<Dataset>) {
    let id = Uuid::new_v4().to_string();

    let (samples, features, labels, task) = match req.kind.as_str() {
        "xor" => (4, 2, 1, "regression"),
        _ => (4, 2, 1, "regression"),
    };

    let dataset = Dataset {
        id: id.clone(),
        name: req.name,
        kind: req.kind,
        samples,
        features,
        labels,
        task: task.to_string(),
        created_at: Utc::now(),
    };

    state.datasets.write().await.insert(id, dataset.clone());
    (StatusCode::CREATED, Json(dataset))
}

pub async fn list_datasets(
    State(state): State<AppState>,
) -> Json<ListDatasetsResponse> {
    let datasets = state.datasets.read().await;
    let datasets_vec: Vec<_> = datasets.values().cloned().collect();
    Json(ListDatasetsResponse {
        datasets: datasets_vec,
    })
}

pub async fn start_training(
    State(state): State<AppState>,
    Json(req): Json<StartTrainingRequest>,
) -> Result<(StatusCode, Json<StartTrainingResponse>), (StatusCode, Json<ErrorResponse>)> {
    let networks = state.networks.read().await;
    let _network = networks.get(&req.network_id)
        .ok_or((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Network not found".to_string(),
            }),
        ))?;
    drop(networks);

    let datasets = state.datasets.read().await;
    let _dataset = datasets.get(&req.dataset_id)
        .ok_or((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Dataset not found".to_string(),
            }),
        ))?;
    drop(datasets);

    let training_id = Uuid::new_v4().to_string();
    let training_state = Arc::new(tokio::sync::RwLock::new(TrainingState {
        running: true,
        total_epochs: req.config.epochs,
        ..Default::default()
    }));

    state.trainers.write().await.insert(training_id.clone(), training_state.clone());

    let training_id_clone = training_id.clone();
    tokio::spawn(async move {
        run_training(training_id_clone, req, training_state).await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(StartTrainingResponse {
            training_id,
            status: "running".to_string(),
        }),
    ))
}

pub async fn get_training_status(
    State(state): State<AppState>,
    Path(training_id): Path<String>,
) -> Result<Json<TrainingStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let trainers = state.trainers.read().await;

    let training_state = trainers.get(&training_id)
        .ok_or((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Training session not found".to_string(),
            }),
        ))?
        .clone();

    drop(trainers);

    let state_guard = training_state.read().await;

    let status = if state_guard.error.is_some() {
        "error".to_string()
    } else if state_guard.stopped {
        "completed".to_string()
    } else if state_guard.paused {
        "paused".to_string()
    } else {
        "running".to_string()
    };

    Ok(Json(TrainingStatusResponse {
        training_id,
        status,
        epoch: state_guard.epoch,
        total_epochs: state_guard.total_epochs,
        last_loss: state_guard.last_loss,
        last_val_loss: state_guard.last_val_loss,
        last_accuracy: state_guard.last_accuracy,
        loss_history: state_guard.loss_history.clone(),
        val_loss_history: state_guard.val_loss_history.clone(),
        accuracy_history: state_guard.accuracy_history.clone(),
        elapsed_secs: state_guard.elapsed_secs,
    }))
}

pub async fn training_websocket(
    State(state): State<AppState>,
    Path(training_id): Path<String>,
    ws: WebSocketUpgrade,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let trainers = state.trainers.read().await;

    let training_state = trainers.get(&training_id)
        .ok_or((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Training session not found".to_string(),
            }),
        ))?
        .clone();

    Ok(ws.on_upgrade(|socket| ws::handle_training_socket(socket, training_state)))
}

async fn run_training(
    _training_id: String,
    req: StartTrainingRequest,
    training_state: Arc<tokio::sync::RwLock<TrainingState>>,
) {
    let layer_specs = vec![
        LayerSpec::Linear { in_dim: 2, out_dim: 8 },
        LayerSpec::Activation(Activation::Tanh),
        LayerSpec::Linear { in_dim: 8, out_dim: 1 },
        LayerSpec::Activation(Activation::Sigmoid),
    ];

    let mut model = Model::from_specs(2, &layer_specs, req.config.seed);

    let optimizer_kind = match req.config.optimizer.kind.as_str() {
        "adam" => OptimizerKind::Adam {
            lr: req.config.optimizer.lr,
            beta1: req.config.optimizer.beta1.unwrap_or(0.9),
            beta2: req.config.optimizer.beta2.unwrap_or(0.999),
            eps: req.config.optimizer.eps.unwrap_or(1e-8),
        },
        "sgd" => OptimizerKind::Sgd {
            lr: req.config.optimizer.lr,
            momentum: 0.9,
        },
        _ => OptimizerKind::Adam {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        },
    };

    let mut optimizer = neuralcabin_engine::optimizer::Optimizer::new(
        optimizer_kind,
        &model.parameter_shapes(),
    );

    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);

    let loss = match req.config.loss.as_str() {
        "mse" => Loss::MeanSquaredError,
        "crossentropy" => Loss::CrossEntropy,
        _ => Loss::MeanSquaredError,
    };

    let start = std::time::Instant::now();

    for epoch in 0..req.config.epochs {
        let l = model.train_step(&mut optimizer, loss, &x, &y);

        {
            let mut state = training_state.write().await;
            state.epoch = epoch + 1;
            state.last_loss = l;
            state.loss_history.push(l);
            state.elapsed_secs = start.elapsed().as_secs_f32();
        }

        if !l.is_finite() {
            let mut state = training_state.write().await;
            state.error = Some(format!("loss diverged to {l} at epoch {epoch}"));
            state.running = false;
            state.stopped = true;
            return;
        }

        tokio::task::yield_now().await;
    }

    let mut state = training_state.write().await;
    state.running = false;
    state.stopped = true;
    state.elapsed_secs = start.elapsed().as_secs_f32();
}
