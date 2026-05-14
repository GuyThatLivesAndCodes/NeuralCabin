use axum::extract::ws::WebSocket;
use std::sync::Arc;
use tokio::time::{interval, Duration};

use crate::models::{TrainingState, WsMessage};

pub async fn handle_training_socket(
    mut socket: WebSocket,
    training_state: Arc<tokio::sync::RwLock<TrainingState>>,
) {
    let mut interval = interval(Duration::from_millis(500));
    let mut last_epoch = 0;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let state = training_state.read().await;

                // Send update if epoch changed
                if state.epoch > last_epoch || state.stopped {
                    let msg = if state.stopped {
                        if let Some(ref error) = state.error {
                            WsMessage::Error {
                                message: error.clone(),
                            }
                        } else {
                            WsMessage::TrainingFinished {
                                status: "completed".to_string(),
                                final_loss: state.last_loss,
                                total_epochs: state.total_epochs,
                                elapsed_secs: state.elapsed_secs,
                            }
                        }
                    } else {
                        WsMessage::EpochUpdate {
                            epoch: state.epoch,
                            total_epochs: state.total_epochs,
                            last_loss: state.last_loss,
                            last_val_loss: state.last_val_loss,
                            last_accuracy: state.last_accuracy,
                            loss_history: state.loss_history.clone(),
                            val_loss_history: state.val_loss_history.clone(),
                            accuracy_history: state.accuracy_history.clone(),
                            elapsed_secs: state.elapsed_secs,
                        }
                    };

                    if let Ok(json) = serde_json::to_string(&msg) {
                        if let Err(e) = socket.send(axum::extract::ws::Message::Text(json)).await {
                            tracing::error!("Failed to send WS message: {}", e);
                            break;
                        }
                    }

                    last_epoch = state.epoch;

                    if state.stopped {
                        break;
                    }
                }
            }
            Some(result) = socket.recv() => {
                match result {
                    Ok(axum::extract::ws::Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
        }
    }

    let _ = socket.close().await;
}
