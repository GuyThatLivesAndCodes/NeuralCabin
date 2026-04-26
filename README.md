# NeuralCabin

NeuralCabin is a hybrid Electron/Rust neural network workbench — build, train, and deploy networks from the ground up with no external ML frameworks. The Rust core handles all compute-heavy operations; the JavaScript UI and autograd graph remain fully intact as the compatibility layer.

## Architecture overview

| Layer | What lives here |
|---|---|
| `src/engine/tensor-js.js` | Pure-JS fallback (autograd, all ops) — always available |
| `src/engine/tensor.js` | Backend selector — picks Rust or JS at startup |
| `native/rust-engine/neuralcabin-core/` | Rust crate: CPU kernels, optimizers, RL, neuroevolution |
| `native/rust-engine/neuralcabin-node/` | N-API bridge exposing Rust to Node.js |
| `src/engine/rl.js` | Q-Learning / DQN agent (JS wrapper over Rust math) |
| `src/engine/neuroevolution.js` | Selective Reproduction / population evolution |
| `src/main/` | Electron main process, training manager, production API server |
| `src/dsl/` | NeuralScript: lexer → parser → type-checker → interpreter |
| `native/cpp-inference-server/` | Optional C++ inference server with AES-256-GCM weight storage |

## Quick start

```bash
npm install
npm start      # Electron app
npm test       # run test suite
```

## Building the Rust engine

Requires Rust stable (≥ 1.78) and Cargo.

```bash
# Check the core crate compiles cleanly
npm run engine:check:rust

# Build the N-API .node binding (release, placed next to index.js)
npm run engine:build:rust
```

On Windows the output will be `neuralcabin-node.win32-x64-msvc.node`. On Linux it will be `neuralcabin-node.linux-x64-gnu.node`. The `index.js` loader tries each automatically.

## Building the C++ inference server (optional)

```bash
npm run server:build:cpp
```

Requires CMake ≥ 3.20 and a C++17 compiler.

## Backend selection

`src/engine/tensor.js` is a runtime backend selector.

| `NEURALCABIN_ENGINE_BACKEND` | Behaviour |
|---|---|
| `auto` (default) | Use Rust binding if available, otherwise fall back to JS |
| `js` | Force the pure-JavaScript engine |
| `rust` | Require the Rust binding; throws on startup if missing |

Optional override for the binding path:

```
NEURALCABIN_NATIVE_BINDING=/absolute/path/to/neuralcabin_node.node
```

## Rust core modules

### `neuralcabin-core/src/cpu.rs` — tensor ops

All operations are parallelised with Rayon and use AVX2 FMA when available.

| Function | Description |
|---|---|
| `add`, `sub`, `mul`, `mul_scalar` | Element-wise arithmetic (with bias-broadcast for `add`) |
| `matmul` | Tiled GEMM (i-k-j loop order, row-aliased writes) |
| `relu`, `leaky_relu`, `tanh_el`, `sigmoid_el`, `gelu_el` | Activations; `gelu_el` returns `(output, tcache)` for grad-free backward |
| `softmax` | Row-wise 2D softmax |
| `softmax_cross_entropy` | Fused log-softmax + NLL; returns `(loss, probs_cache)` |
| `mse_loss` | Mean squared error scalar |
| `dropout` | Inverted dropout with Mulberry32 mask; returns `(output, mask)` |
| `embedding` | Token lookup `[V,D] × [B] → [B,D]` |
| `sum_all`, `randn` | Reduction and random normal init |
| `has_nan_or_inf` | Fast NaN/Inf guard |

### `neuralcabin-core/src/layers.rs` — layer forward passes

- `linear_forward(input, weight, bias)` — `[B,in] @ [out,in]^T + [out]`
- `embedding_forward` / `embedding_backward` — scatter-add gradient accumulation
- `sequential_forward_inference` — chain of `LayerDesc` variants for pure-Rust inference

### `neuralcabin-core/src/optim.rs` — optimizer steps

All functions are stateless (state arrays passed in/out) so the JS side owns serialization.

| Function | Description |
|---|---|
| `sgd_step` | SGD with momentum and L2 weight decay |
| `adam_step` | Adam with bias correction |
| `adamw_step` | AdamW (decoupled weight decay) |
| `clip_grad_norm` | Global gradient clipping |

### `neuralcabin-core/src/rl.rs` — Q-Learning / DQN

| Struct / Function | Description |
|---|---|
| `ReplayBuffer` | Flat circular buffer for `(s, a, r, s', done)` transitions |
| `epsilon_greedy` | ε-greedy action selection |
| `compute_td_targets` | `r + γ · max Q(s')` Bellman target |
| `dqn_loss` / `dqn_huber_loss` | MSE / Huber loss over chosen-action Q values + gradient |
| `soft_update_target` | Polyak averaging: `θ_target ← (1−τ)θ_target + τθ_online` |

### `neuralcabin-core/src/neuroevolution.rs` — Selective Reproduction

| Function | Description |
|---|---|
| `mutate` | Per-weight Gaussian perturbation |
| `crossover_uniform` / `crossover_single_point` / `crossover_arithmetic` | Three crossover strategies |
| `tournament_select` / `roulette_select` / `truncation_select` | Three selection strategies |
| `evolve_generation` | One full generation: elitism → parent selection → crossover → mutation |
| `fitness_stats` | Returns `(min, max, mean, std)` |

## JavaScript API

### Q-Learning (`src/engine/rl.js`)

```js
const { DQNAgent, ReplayBuffer } = require('./src/engine/rl');

const agent = new DQNAgent({
  architecture: { kind: 'classifier', inputDim: 4, outputDim: 2, hidden: [64, 64] },
  gamma: 0.99, lr: 1e-3, batchSize: 64,
  epsilonStart: 1.0, epsilonEnd: 0.05, epsilonDecay: 0.995,
  targetUpdateFreq: 100, seed: 42,
});

// Episode loop
const action = agent.selectAction(state);           // ε-greedy
agent.observe(state, action, reward, nextState, done);
const loss = agent.trainStep();                      // null until buffer is ready

const saved = agent.toJSON();
const restored = DQNAgent.fromJSON(saved);
```

### Neuroevolution (`src/engine/neuroevolution.js`)

```js
const { Population, evolveNetwork } = require('./src/engine/neuroevolution');

const pop = new Population({
  architecture: { kind: 'classifier', inputDim: 4, outputDim: 2, hidden: [32] },
  size: 50, eliteCount: 5, pMutate: 0.1, mutationStd: 0.02,
});

// Manual loop
pop.evaluate((model, i) => runEpisode(model));
const stats = pop.evolve(); // { min, max, mean, std }

// Or use the async convenience wrapper
const result = await evolveNetwork(pop, 100, (model) => runEpisode(model), {
  onGeneration: ({ generation, stats }) => console.log(generation, stats),
  shouldStop: () => earlyStop,
});
```

## Production API server

The HTTP inference server (`src/main/api-server.js`) now includes:

- **Rate limiting** — sliding-window per IP, configurable via `opts.rateLimit` (requests/minute, default 120)
- **Auth** — HS256-signed Bearer tokens. Pass `opts.authSecret` to `server.start()` to enable; call `server.issueToken(id)` to mint a token
- **`/metrics`** — Prometheus text format with request counts, error counts, avg latency, and uptime
- **`/health`** — `{ ok: true, uptime }` (not rate-limited, not auth-gated)
- **`/chat/reset`** — clear a named session without creating a new one

```js
// Start with auth + rate limiting
const url = await apiServer.start(networkId, 8080, {
  rateLimit: 30,            // requests per minute per IP
  authSecret: 'my-secret',  // omit for open access
});

const token = apiServer.issueToken(networkId, 3600); // expires in 1h
// curl -H "Authorization: Bearer <token>" http://localhost:8080/predict ...
```

## Project layout

```text
src/
  engine/
    tensor.js                backend selector
    tensor-js.js             pure-JS fallback engine (autograd, all ops)
    tensor-native-loader.js  N-API loader with compatibility check
    trainer.js               training / inference loop
    optim.js                 10 optimizer implementations
    rl.js                    Q-Learning / DQN agent
    neuroevolution.js        Selective Reproduction / population evolution
    model.js, layers.js      model builders
    tokenizer.js, chat-format.js
  dsl/                       NeuralScript (lexer, parser, typecheck, interpreter)
  main/                      Electron main, TrainingManager, ApiServer, Storage
  renderer/                  UI
  plugins/                   plugin system
native/
  rust-engine/
    neuralcabin-core/        Rust library (cpu, layers, optim, rl, neuroevolution, …)
    neuralcabin-node/        N-API bridge (src/lib.rs + index.js)
  cpp-inference-server/      C++ inference scaffold
tests/
  run-tests.js
docs/
  rust-migration.md
```
