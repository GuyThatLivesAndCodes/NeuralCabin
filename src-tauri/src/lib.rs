mod models;

use models::*;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use chrono::Utc;
use neuralcabin_engine::nn::{LayerSpec, Model};
use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
use neuralcabin_engine::tensor::Tensor;
use neuralcabin_engine::{Activation, Loss};

pub struct AppState {
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

#[tauri::command]
async fn create_network(
    state: State<'_, AppState>,
    req: CreateNetworkRequest,
) -> Result<Network, String> {
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
    Ok(network)
}

#[tauri::command]
async fn list_networks(state: State<'_, AppState>) -> Result<Vec<Network>, String> {
    Ok(state.networks.read().await.values().cloned().collect())
}

#[tauri::command]
async fn get_network(state: State<'_, AppState>, id: String) -> Result<Network, String> {
    state
        .networks
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| "Network not found".to_string())
}

#[tauri::command]
async fn delete_network(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    Ok(state.networks.write().await.remove(&id).is_some())
}

#[tauri::command]
async fn create_dataset(
    state: State<'_, AppState>,
    req: CreateDatasetRequest,
) -> Result<Dataset, String> {
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
    Ok(dataset)
}

#[tauri::command]
async fn list_datasets(state: State<'_, AppState>) -> Result<Vec<Dataset>, String> {
    Ok(state.datasets.read().await.values().cloned().collect())
}

#[tauri::command]
async fn start_training(
    app: AppHandle,
    state: State<'_, AppState>,
    req: StartTrainingRequest,
) -> Result<StartTrainingResponse, String> {
    if !state.networks.read().await.contains_key(&req.network_id) {
        return Err("Network not found".to_string());
    }
    if !state.datasets.read().await.contains_key(&req.dataset_id) {
        return Err("Dataset not found".to_string());
    }

    let training_id = Uuid::new_v4().to_string();
    let training_state = Arc::new(RwLock::new(TrainingState {
        running: true,
        total_epochs: req.config.epochs,
        ..Default::default()
    }));

    state
        .trainers
        .write()
        .await
        .insert(training_id.clone(), training_state.clone());

    let id_clone = training_id.clone();
    tokio::spawn(async move {
        run_training(app, id_clone, req, training_state).await;
    });

    Ok(StartTrainingResponse {
        training_id,
        status: "running".to_string(),
    })
}

#[tauri::command]
async fn get_training_status(
    state: State<'_, AppState>,
    training_id: String,
) -> Result<TrainingStatusResponse, String> {
    let trainers = state.trainers.read().await;
    let ts = trainers
        .get(&training_id)
        .ok_or_else(|| "Training session not found".to_string())?
        .clone();
    drop(trainers);

    let s = ts.read().await;
    let status = if s.error.is_some() {
        "error"
    } else if s.stopped {
        "completed"
    } else {
        "running"
    };

    Ok(TrainingStatusResponse {
        training_id,
        status: status.to_string(),
        epoch: s.epoch,
        total_epochs: s.total_epochs,
        last_loss: s.last_loss,
        loss_history: s.loss_history.clone(),
        elapsed_secs: s.elapsed_secs,
    })
}

async fn run_training(
    app: AppHandle,
    training_id: String,
    req: StartTrainingRequest,
    training_state: Arc<RwLock<TrainingState>>,
) {
    let layer_specs = vec![
        LayerSpec::Linear { in_dim: 2, out_dim: 8 },
        LayerSpec::Activation(Activation::Tanh),
        LayerSpec::Linear { in_dim: 8, out_dim: 1 },
        LayerSpec::Activation(Activation::Sigmoid),
    ];

    let mut model = Model::from_specs(2, &layer_specs, req.config.seed);

    let optimizer_kind = match req.config.optimizer.kind.as_str() {
        "sgd" => OptimizerKind::Sgd {
            lr: req.config.optimizer.lr,
            momentum: req.config.optimizer.momentum.unwrap_or(0.9),
        },
        _ => OptimizerKind::Adam {
            lr: req.config.optimizer.lr,
            beta1: req.config.optimizer.beta1.unwrap_or(0.9),
            beta2: req.config.optimizer.beta2.unwrap_or(0.999),
            eps: req.config.optimizer.eps.unwrap_or(1e-8),
        },
    };

    let mut optimizer = Optimizer::new(optimizer_kind, &model.parameter_shapes());

    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);

    let loss_fn = match req.config.loss.as_str() {
        "crossentropy" => Loss::CrossEntropy,
        _ => Loss::MeanSquaredError,
    };

    let start = std::time::Instant::now();
    let mut loss_history: Vec<f32> = Vec::with_capacity(req.config.epochs);

    for epoch in 1..=req.config.epochs {
        let loss = model.train_step(&mut optimizer, loss_fn, &x, &y);
        loss_history.push(loss);

        let elapsed = start.elapsed().as_secs_f32();
        {
            let mut s = training_state.write().await;
            s.epoch = epoch;
            s.last_loss = loss;
            s.loss_history = loss_history.clone();
            s.elapsed_secs = elapsed;
        }

        let _ = app.emit(
            "training_update",
            TrainingUpdate {
                training_id: training_id.clone(),
                epoch,
                total_epochs: req.config.epochs,
                loss,
                loss_history: loss_history.clone(),
                elapsed_secs: elapsed,
            },
        );

        if !loss.is_finite() {
            let mut s = training_state.write().await;
            s.error = Some(format!("loss diverged at epoch {epoch}"));
            s.running = false;
            s.stopped = true;
            let _ = app.emit(
                "training_error",
                TrainingError {
                    training_id,
                    message: format!("Loss diverged to {loss} at epoch {epoch}"),
                },
            );
            return;
        }

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

    let _ = app.emit(
        "training_finished",
        TrainingFinished {
            training_id,
            status: "completed".to_string(),
            final_loss,
            total_epochs: req.config.epochs,
            elapsed_secs: elapsed,
        },
    );
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            create_network,
            list_networks,
            get_network,
            delete_network,
            create_dataset,
            list_datasets,
            start_training,
            get_training_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
