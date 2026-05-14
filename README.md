# 🧠 NeuralCabin

A **pure Rust neural network workbench** with a **modern web-based UI**.

The workbench consists of two parts:

- **Backend:** Pure Rust REST/WebSocket API server (no Python, PyTorch, TensorFlow, NumPy, etc.)
- **Frontend:** React + TypeScript web application

The neural-network engine has **zero external math dependencies**: tensors,
matmul, activations, losses, autograd and optimisers are all hand-written in pure Rust.

## Quick Start

### Download & run (pre-built binary)
1. Download the binary for your OS from the [Actions tab](../../actions) or [Releases](../../releases)
2. Extract the archive and run `neuralcabin` (Linux/macOS) or `neuralcabin.exe` (Windows)
3. Your browser opens automatically at `http://localhost:3001`

### Build from source
```bash
# 1. Build the React UI
cd frontend && npm install && npm run build && cd ..

# 2. Build the Rust server (frontend is embedded inside)
cargo build --package neuralcabin-backend --release

# 3. Run — opens browser automatically
./target/release/neuralcabin
```

Or use the convenience script:
```bash
./start-dev.sh        # Linux/macOS
start-dev.bat         # Windows
```

## Repository Layout

```
neuralcabin/
├── Cargo.toml              — workspace manifest
├── src/main.rs             — entry point (headless mode only; UI moved to web)
│
├── engine/                 — Pure Rust ML engine (zero math deps)
│   └── src/
│       ├── tensor.rs       — Dense Vec<f32> tensors, matmul, operations
│       ├── autograd.rs     — Reverse-mode autodiff (tape-based)
│       ├── activations.rs  — ReLU, Sigmoid, Tanh, Softmax
│       ├── loss.rs         — MSE, CrossEntropy
│       ├── optimizer.rs    — SGD (momentum), Adam
│       ├── nn.rs           — Linear/Activation layers, Model
│       └── persistence.rs  — Model checkpoints (JSON)
│
├── backend/                — Rust REST/WebSocket API (Axum)
│   └── src/
│       ├── main.rs         — Server entry point (port 3001)
│       ├── handlers.rs     — REST endpoint handlers
│       ├── models.rs       — Request/response types
│       └── ws.rs           — WebSocket real-time updates
│
├── frontend/               — React + TypeScript web UI
│   ├── src/
│   │   ├── App.tsx         — Main 7-tab application
│   │   ├── api.ts          — Axios API client
│   │   ├── index.css       — Styling (Times New Roman, orange theme)
│   │   └── tabs/           — Tab components
│   ├── package.json
│   ├── vite.config.ts
│   └── index.html
│
├── ui/                     — DEPRECATED: Old egui desktop UI (kept for reference)
│
├── API_CONTRACT.md         — REST/WebSocket API specification
├── REFACTOR_README.md      — Architecture documentation
├── QUICKSTART.md           — Quick start guide
├── DELIVERY_SUMMARY.md     — Project completion summary
└── start-dev.sh/bat        — Convenience startup scripts
```

## Building & Running

### How it works

The React frontend is compiled and **embedded directly inside the Rust binary** at build time using `rust-embed`. So you get a single self-contained executable — no separate web server, no Node.js needed at runtime.

```
neuralcabin (single binary)
  ├── Serves the React web UI at http://localhost:3001/
  ├── REST API at /api/...
  └── WebSocket at /ws/...
```

### Development Workflow

```bash
# Rebuild frontend + restart server in one command:
./start-dev.sh           # Linux/macOS
start-dev.bat            # Windows

# Or manually:
cd frontend && npm run build && cd ..
cargo run --package neuralcabin-backend --release
```

### Release Build

```bash
cd frontend && npm ci && npm run build && cd ..
cargo build --package neuralcabin-backend --release
# Output: target/release/neuralcabin  (self-contained, no dependencies)
```

### Headless Testing

```bash
# Run XOR training demo (no UI)
cargo run --release -- --xor-demo

# Show help
cargo run -- --help
```

## Features

### ✨ Web-Based UI (React + TypeScript)

- **7 Tabs:** Docs, Networks, Corpus, Vocab, Training, Inference, Plugins
- **Orange Theme:** Warm, inviting design with Times New Roman typography
- **Real-time Training:** Live loss plotting via WebSocket
- **Responsive:** Works on desktop, tablet, and mobile
- **Modern Stack:** React, TypeScript, Vite, Axios

### 🚀 REST/WebSocket API

- **REST Endpoints:** Create/list/delete networks and datasets, start training
- **WebSocket:** Real-time training metrics (loss, epoch, time, etc.)
- **Type-Safe:** Serde serialization, TypeScript types on frontend

### 🧠 Pure Rust Engine

- **Zero Math Dependencies:** Tensors, matmul, autograd, optimizers all hand-written
- **Fast:** Optimized Rust with no Python overhead
- **Flexible:** Supports multiple optimizers (SGD, Adam) and loss functions

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** — Get running in 5 minutes
- **[API_CONTRACT.md](API_CONTRACT.md)** — Complete API specification
- **[REFACTOR_README.md](REFACTOR_README.md)** — Architecture deep dive
- **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** — Project overview

## Tests

```bash
cargo test --workspace
```

The engine ships with 13 tests including:

- analytic-vs-numerical gradient check on `MatMul + MSE`,
- standalone `Sigmoid` backward correctness,
- SGD and Adam convergence on a quadratic,
- CSV parser (with and without one-hot encoding),
- model save/load round-trip,
- end-to-end MLP convergence on XOR.

`cargo clippy --workspace --all-targets -- -D warnings` is also clean.

## Using NeuralCabin

### Networks Tab
Create and manage neural network architectures. Specify layer counts, types, and initialization seeds.

### Training Tab
Configure and run training sessions:
- Select a network and dataset
- Choose optimizer (Adam, SGD) and loss function (MSE, CrossEntropy)
- Monitor real-time loss plot as the network trains
- Training metrics update every epoch via WebSocket

### Documentation Tab
In-app guide with:
- Step-by-step getting started instructions
- Feature overview for each tab
- Technology stack information
- Current features and roadmap

### Coming Soon
- Corpus/dataset management
- Vocabulary editor
- Inference/generation features
- Plugin system

## Engine quick reference

```rust
use neuralcabin_engine::{
    nn::{LayerSpec, Model},
    optimizer::{Optimizer, OptimizerKind},
    tensor::Tensor,
    Activation, Loss,
};

let mut model = Model::from_specs(2, &[
    LayerSpec::Linear { in_dim: 2, out_dim: 8 },
    LayerSpec::Activation(Activation::Tanh),
    LayerSpec::Linear { in_dim: 8, out_dim: 1 },
    LayerSpec::Activation(Activation::Sigmoid),
], 42);
let mut opt = Optimizer::new(
    OptimizerKind::Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
    &model.parameter_shapes(),
);
let x = Tensor::new(vec![4, 2], vec![0.,0., 0.,1., 1.,0., 1.,1.]);
let y = Tensor::new(vec![4, 1], vec![0., 1., 1., 0.]);
for _ in 0..2000 {
    model.train_step(&mut opt, Loss::MeanSquaredError, &x, &y);
}
let pred = model.predict(&x);
```

## License

MIT — see `LICENSE.md`.
