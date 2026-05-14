# NeuralCabin: Rust Backend + React Frontend Architecture

This document describes the refactored NeuralCabin architecture, which separates the monolithic single binary into a REST/WebSocket backend and React frontend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  React Frontend (TypeScript)                            │
│  - 7 Tabs: Docs, Networks, Corpus, Vocab, Training,    │
│    Inference, Plugins                                   │
│  - Real-time loss plotting via WebSocket               │
│  - Axios for REST API calls                            │
└─────────────┬───────────────────────────────────────────┘
              │
         REST / WebSocket
         (port 3001)
              │
┌─────────────▼───────────────────────────────────────────┐
│  Rust Backend (Axum)                                    │
│  - REST endpoints for CRUD operations                  │
│  - WebSocket endpoint for training updates              │
│  - Training loop with shared state                      │
└─────────────┬───────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│  NeuralCabin Engine (Pure Rust)                        │
│  - Tensor operations (unchanged)                        │
│  - Autograd system (unchanged)                          │
│  - Neural network layers (unchanged)                    │
│  - Optimizers: SGD, Adam, AdamW, LAMB                  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Rust 1.70+
- Node.js 18+
- npm or yarn

### Running the Backend

```bash
cd /home/user/NeuralCabin
cargo run --package neuralcabin-backend --release
```

The backend will start on `http://127.0.0.1:3001`.

### Running the Frontend

```bash
cd /home/user/NeuralCabin/frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`.

## API Contract

See `API_CONTRACT.md` for complete API specification.

### Quick API Examples

#### Create a Network
```bash
curl -X POST http://localhost:3001/api/networks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "xor-mlp",
    "kind": "simplex",
    "seed": 42,
    "layers": [
      {"type": "linear", "in_dim": 2, "out_dim": 8},
      {"type": "activation", "activation": "tanh"},
      {"type": "linear", "in_dim": 8, "out_dim": 1},
      {"type": "activation", "activation": "sigmoid"}
    ]
  }'
```

#### Create a Dataset
```bash
curl -X POST http://localhost:3001/api/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "xor",
    "kind": "xor",
    "seed": 42
  }'
```

#### Start Training
```bash
curl -X POST http://localhost:3001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "network_id": "<network-id>",
    "dataset_id": "<dataset-id>",
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
  }'
```

#### Connect to Training WebSocket
```javascript
const ws = new WebSocket('ws://localhost:3001/ws/train/<training-id>')

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data)
  console.log(`Epoch ${msg.epoch}/${msg.total_epochs}: loss=${msg.last_loss}`)
}
```

## End-to-End Flow

1. **User opens the app** → React frontend loads
2. **User creates a network** → Frontend calls `POST /api/networks`
3. **Backend stores network** → In-memory HashMap
4. **User creates a dataset** → Frontend calls `POST /api/datasets`
5. **Backend creates dataset** → In-memory HashMap
6. **User starts training** → Frontend calls `POST /api/train`
7. **Backend spawns training task** → Tokio async task
8. **Frontend connects WebSocket** → `ws://localhost:3001/ws/train/<training-id>`
9. **Training runs** → Updates shared state every epoch
10. **WebSocket sends updates** → JSON messages with loss, epoch, etc.
11. **Frontend receives updates** → Updates React state, re-renders plot
12. **Training completes** → WebSocket sends `training_finished` message

## Project Structure

```
NeuralCabin/
├── engine/                    # ML engine library (unchanged)
│   ├── src/
│   │   ├── tensor.rs
│   │   ├── autograd.rs
│   │   ├── nn.rs
│   │   ├── optimizer.rs
│   │   ├── loss.rs
│   │   ├── activations.rs
│   │   └── ...
│   └── Cargo.toml
├── backend/                   # REST/WebSocket backend (NEW)
│   ├── src/
│   │   ├── main.rs          # Server entry point
│   │   ├── handlers.rs      # REST endpoint handlers
│   │   ├── models.rs        # Request/response types
│   │   └── ws.rs            # WebSocket handler
│   └── Cargo.toml
├── frontend/                  # React frontend (NEW)
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── api.ts           # API client
│   │   ├── tabs/
│   │   │   ├── NetworksTab.tsx
│   │   │   ├── TrainingTab.tsx
│   │   │   ├── ...
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── package.json
├── ui/                        # Old egui UI (kept for reference)
├── src/
│   └── main.rs              # Binary that previously tied UI + engine
├── Cargo.toml               # Workspace manifest (updated)
├── API_CONTRACT.md          # API specification
└── REFACTOR_README.md       # This file
```

## Key Design Decisions

### 1. **In-Memory Storage (MVP)**
Currently, networks, datasets, and training sessions are stored in HashMaps in memory.
- **Pro:** Simple, fast, no DB complexity for MVP
- **Con:** Data is lost on restart
- **Future:** Replace with persistent storage (file system, database)

### 2. **Async Training Loop**
The training loop runs in a Tokio async task spawned by the training endpoint.
- **Pro:** Non-blocking, multiple trainings can run in parallel
- **Con:** No pause/resume yet (only stop)
- **Future:** Add pause/resume commands via channels

### 3. **Shared State via RwLock**
Training state is wrapped in `Arc<RwLock<TrainingState>>` for safe sharing.
- **Pro:** Thread-safe, allows both read and write access
- **Con:** Slight overhead
- **Future:** Consider Parking Lot's RwLock for better performance

### 4. **WebSocket Polling**
The WebSocket handler polls the training state every 500ms and sends updates if the epoch changed.
- **Pro:** Simple, avoids tight loops
- **Con:** Updates are not instantaneous
- **Future:** Use channels for more immediate updates

### 5. **Fixed Architecture for MVP**
The network architecture is hardcoded to `2 → 8 (tanh) → 1 (sigmoid)` for simplicity.
- **Pro:** Minimal code, focuses on API architecture
- **Con:** Not flexible
- **Future:** Build network from layer specs sent in the request

## Testing

### Manual Testing Checklist

- [ ] Backend starts on port 3001
- [ ] Frontend starts on port 5173
- [ ] Can create a network via `/api/networks`
- [ ] Can create a dataset via `/api/datasets`
- [ ] Can start training via `/api/train`
- [ ] WebSocket receives epoch updates during training
- [ ] Loss plot updates in real-time in the frontend
- [ ] Training completes and shows "completed" message
- [ ] Can start new training after first one completes

### Running Tests

```bash
# Test backend
cargo test --package neuralcabin-backend

# Test engine (unchanged)
cargo test --package neuralcabin-engine

# Test frontend
cd frontend && npm test
```

## Future Work

### Phase 2: Core Features
- [ ] Persist networks and datasets to disk
- [ ] Implement pause/resume training
- [ ] Build network from arbitrary layer specs
- [ ] Add inference endpoint
- [ ] Add validation loss tracking
- [ ] Support multiple optimizers in UI

### Phase 3: Advanced Features
- [ ] Plugins system
- [ ] Corpus management
- [ ] Vocabulary editor
- [ ] Model export/import
- [ ] Training resumption from checkpoint
- [ ] Batch size tuning

### Phase 4: Polish
- [ ] Better error messages
- [ ] Progress indicators
- [ ] Training history/analytics
- [ ] Keyboard shortcuts
- [ ] Dark/light theme toggle
- [ ] Performance optimizations

## Troubleshooting

### Backend fails to start
```
error: Address already in use (os error 48)
```
**Solution:** Port 3001 is in use. Kill the process or change the port in `backend/src/main.rs`.

### WebSocket connection fails
**Problem:** `WebSocket connection error`
**Solution:** 
1. Ensure backend is running on port 3001
2. Check browser console for CORS/mixed content errors
3. Verify training ID is valid

### Frontend shows "Network not found"
**Problem:** Network list is empty when creating dataset
**Solution:**
1. Refresh the page
2. Create a network first in the Networks tab
3. Wait for list to load (check browser DevTools Network tab)

## Contributing

When making changes:
1. Update `API_CONTRACT.md` if API changes
2. Keep engine code unchanged (focus on backend/frontend)
3. Test end-to-end flow (create network → train → see loss plot)
4. Commit both backend and frontend changes together

## License

MIT (same as NeuralCabin)
