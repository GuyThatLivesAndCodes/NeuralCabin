# NeuralCabin Refactored - Quick Start Guide

This guide walks you through getting the new Rust backend + React frontend architecture running.

## Prerequisites

Ensure you have installed:
- **Rust** 1.70 or later ([install](https://rustup.rs/))
- **Node.js** 18 or later ([install](https://nodejs.org/))
- **npm** (comes with Node.js)

## Setup (One-Time)

### 1. Install Backend Dependencies
```bash
cd /home/user/NeuralCabin
cargo build --package neuralcabin-backend --release
```

### 2. Install Frontend Dependencies
```bash
cd /home/user/NeuralCabin/frontend
npm install
```

## Running the Application

### Option A: Manual (Two Terminals)

**Terminal 1 - Start the Backend:**
```bash
cd /home/user/NeuralCabin
cargo run --package neuralcabin-backend --release
```
Expected output:
```
2025-05-14T10:30:00Z INFO  NeuralCabin backend server running on http://127.0.0.1:3001
```

**Terminal 2 - Start the Frontend:**
```bash
cd /home/user/NeuralCabin/frontend
npm run dev
```
Expected output:
```
  VITE v5.0.0  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

Then open http://localhost:5173/ in your browser.

### Option B: Using Helper Script (Requires both to be in PATH)

We'll create a convenient startup script:

```bash
#!/bin/bash
# Save this as ~/start-neuralcabin.sh

cd ~/NeuralCabin

# Start backend in background
echo "Starting backend..."
cargo run --package neuralcabin-backend --release &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
```

Make it executable:
```bash
chmod +x ~/start-neuralcabin.sh
```

Then run:
```bash
~/start-neuralcabin.sh
```

## First Time Usage

### Step 1: Create a Network
1. Open the app at http://localhost:5173
2. Click the **Networks** tab
3. Click **Create Network**
4. Fill in the form (defaults are fine):
   - Name: `xor-mlp`
   - Kind: `simplex`
   - Seed: `42`
5. Click **Create**
6. You should see the network listed below

### Step 2: Create a Dataset
1. Click the **Training** tab
2. If no dataset exists, click **Create XOR Dataset**
3. The dataset should appear in the dropdown

### Step 3: Start Training
1. Select the network you created (should be pre-selected)
2. Select the dataset (XOR)
3. Click **Start Training**
4. Watch the loss plot update in real-time as the model trains
5. The progress bar shows epoch progress
6. The loss should decrease over 2000 epochs

### Step 4: Monitor Progress
- The plot shows loss history in real-time
- Elapsed time updates every epoch
- Training completes automatically after 2000 epochs

## Troubleshooting

### Backend fails to start with "Address already in use"
**Problem:** Port 3001 is already in use
**Solution:**
```bash
# Find and kill the process using port 3001
lsof -i :3001
kill -9 <PID>

# Or change port in backend/src/main.rs line ~57
# .bind("127.0.0.1:3002")
```

### Frontend shows "Cannot connect to backend"
**Problem:** Backend is not running or on wrong port
**Solution:**
1. Check backend is running: `curl http://localhost:3001/api/networks`
2. Should return: `{"networks":[]}`
3. If not, restart backend

### WebSocket connection fails
**Problem:** `WebSocket connection error`
**Solution:**
1. Check browser console (F12) for detailed error
2. Ensure backend is running
3. Training must have been started (training_id must be valid)

### Networks list empty
**Problem:** "No networks yet" message in Networks tab
**Solution:**
1. Click "Create Network"
2. Use defaults or set values
3. Click "Create"

### Dataset dropdown shows nothing
**Problem:** No datasets available
**Solution:**
1. In Training tab, click "Create XOR Dataset"
2. Wait for confirmation
3. Refresh page (F5) if needed

### Loss plot not updating
**Problem:** Plot shows "Waiting for training data..."
**Solution:**
1. Click "Start Training"
2. Wait a moment for first epoch to complete
3. Loss should appear on plot
4. If still nothing, check browser console for errors

## Command Reference

### Backend API (for manual testing)

List networks:
```bash
curl http://localhost:3001/api/networks
```

Create network:
```bash
curl -X POST http://localhost:3001/api/networks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test",
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

Create dataset:
```bash
curl -X POST http://localhost:3001/api/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "xor",
    "kind": "xor",
    "seed": 42
  }'
```

Start training:
```bash
curl -X POST http://localhost:3001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "network_id": "550e8400-e29b-41d4-a716-446655440000",
    "dataset_id": "650e8400-e29b-41d4-a716-446655440001",
    "config": {
      "epochs": 2000,
      "batch_size": 4,
      "optimizer": {"kind": "adam", "lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
      "loss": "mse",
      "validation_frac": 0.2,
      "seed": 42
    }
  }'
```

## Next Steps

After successfully running the app:
1. Read `REFACTOR_README.md` for architecture details
2. Review `API_CONTRACT.md` for complete API specification
3. Check `TODO.md` for planned features
4. Start implementing new features!

## Build for Production

### Backend Release Build
```bash
cargo build --package neuralcabin-backend --release
# Binary: target/release/neuralcabin-backend
```

### Frontend Production Build
```bash
cd frontend
npm run build
# Output: frontend/dist/
```

## Development Tips

### Hot Reload
- **Frontend:** Automatic with Vite (just save files)
- **Backend:** Use `cargo watch` for automatic rebuild:
  ```bash
  cargo install cargo-watch
  cargo watch -x "run --package neuralcabin-backend"
  ```

### Debugging
- **Frontend:** Open DevTools (F12) to see Network tab and Console
- **Backend:** Set `RUST_LOG=debug` before running:
  ```bash
  RUST_LOG=debug cargo run --package neuralcabin-backend
  ```

### Testing
```bash
# Backend tests
cargo test --package neuralcabin-backend

# Frontend tests
cd frontend && npm test
```

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review `REFACTOR_README.md` for architecture questions
3. Check `API_CONTRACT.md` for API details
4. Review browser console (F12) for frontend errors
5. Check terminal output for backend errors

## Further Reading

- `REFACTOR_README.md` - Full architecture documentation
- `API_CONTRACT.md` - Complete API specification
- Backend source: `backend/src/`
- Frontend source: `frontend/src/`
