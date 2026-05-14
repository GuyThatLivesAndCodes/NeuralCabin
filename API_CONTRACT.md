# NeuralCabin API Contract

This document defines the REST and WebSocket API endpoints for the NeuralCabin backend server.

## Base URL
```
http://localhost:3001
```

## REST Endpoints

### Networks

#### Create Network
```
POST /api/networks
Content-Type: application/json

{
  "name": "xor-mlp",
  "kind": "simplex",
  "seed": 42,
  "layers": [
    { "type": "linear", "in_dim": 2, "out_dim": 8 },
    { "type": "activation", "activation": "tanh" },
    { "type": "linear", "in_dim": 8, "out_dim": 1 },
    { "type": "activation", "activation": "sigmoid" }
  ]
}

Response: 201 Created
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "xor-mlp",
  "kind": "simplex",
  "layers": [...],
  "created_at": "2025-05-14T10:30:00Z"
}
```

#### List Networks
```
GET /api/networks

Response: 200 OK
{
  "networks": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "xor-mlp",
      "kind": "simplex",
      "layers": [...]
    }
  ]
}
```

#### Get Network
```
GET /api/networks/:id

Response: 200 OK
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "xor-mlp",
  "kind": "simplex",
  "layers": [...]
}
```

#### Delete Network
```
DELETE /api/networks/:id

Response: 204 No Content
```

### Datasets

#### Create Dataset
```
POST /api/datasets
Content-Type: application/json

{
  "name": "xor",
  "kind": "xor",
  "seed": 42
}

Response: 201 Created
{
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "name": "xor",
  "kind": "xor",
  "samples": 4,
  "features": 2,
  "labels": 1,
  "task": "regression"
}
```

#### List Datasets
```
GET /api/datasets

Response: 200 OK
{
  "datasets": [
    {
      "id": "650e8400-e29b-41d4-a716-446655440001",
      "name": "xor",
      "kind": "xor",
      "samples": 4,
      "features": 2,
      "labels": 1,
      "task": "regression"
    }
  ]
}
```

### Training

#### Start Training
```
POST /api/train
Content-Type: application/json

{
  "network_id": "550e8400-e29b-41d4-a716-446655440000",
  "dataset_id": "650e8400-e29b-41d4-a716-446655440001",
  "config": {
    "epochs": 2000,
    "batch_size": 4,
    "optimizer": {
      "kind": "adam",
      "lr": 0.05,
      "beta1": 0.9,
      "beta2": 0.999,
      "eps": 1e-8
    },
    "loss": "mse",
    "validation_frac": 0.2,
    "seed": 42
  }
}

Response: 202 Accepted
{
  "training_id": "750e8400-e29b-41d4-a716-446655440002",
  "status": "running"
}
```

#### Get Training Status
```
GET /api/train/:training_id

Response: 200 OK
{
  "training_id": "750e8400-e29b-41d4-a716-446655440002",
  "status": "running",
  "epoch": 500,
  "total_epochs": 2000,
  "last_loss": 0.001234,
  "last_val_loss": 0.001567,
  "loss_history": [0.5, 0.3, ...],
  "val_loss_history": [0.55, 0.32, ...],
  "elapsed_secs": 10.5
}
```

## WebSocket Endpoint

### Training Updates
```
WS /ws/train/:training_id
```

Connect after starting training to receive real-time updates.

#### Message Format (Server → Client)
```json
{
  "type": "epoch_update",
  "epoch": 1,
  "total_epochs": 2000,
  "last_loss": 0.123456,
  "last_val_loss": 0.125000,
  "last_accuracy": null,
  "loss_history": [0.123456],
  "val_loss_history": [0.125000],
  "accuracy_history": [],
  "elapsed_secs": 0.123
}
```

```json
{
  "type": "training_finished",
  "status": "completed",
  "final_loss": 0.001234,
  "total_epochs": 2000,
  "elapsed_secs": 120.5
}
```

```json
{
  "type": "error",
  "message": "loss diverged to inf at epoch 500"
}
```

## Error Responses

All endpoints follow standard HTTP error codes:

```
400 Bad Request - Invalid input
404 Not Found - Resource not found
500 Internal Server Error - Server error
```

Error response format:
```json
{
  "error": "descriptive error message"
}
```

## Data Types

### Activation
```
"relu" | "tanh" | "sigmoid"
```

### Layer
```
{
  "type": "linear",
  "in_dim": number,
  "out_dim": number
}
|
{
  "type": "activation",
  "activation": "relu" | "tanh" | "sigmoid"
}
```

### OptimizerKind
```
{
  "kind": "adam",
  "lr": number,
  "beta1": number,
  "beta2": number,
  "eps": number
}
|
{
  "kind": "sgd",
  "lr": number
}
```

### Loss
```
"mse" | "crossentropy"
```

### DatasetKind
```
"xor"
```

### NetworkKind
```
"simplex" | "gpt" | "next_token_gen"
```
