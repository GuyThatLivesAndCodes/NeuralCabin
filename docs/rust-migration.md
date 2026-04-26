# NeuralCabin Rust/WebGPU Migration Notes

## Goals

1. Replace `src/engine/tensor.js` execution with native Rust kernels.
2. Keep the UI/API contract stable during migration.
3. Move NeuralScript validation errors to compile time where possible.

## Stage 1 (implemented in this branch)

- `src/engine/tensor.js` became a backend selector.
- `src/engine/tensor-js.js` keeps full JS behavior as fallback.
- `src/dsl/compiler.js` + `src/dsl/typecheck.js` now gate script execution.
- Added `native/rust-engine/` workspace with:
  - `neuralcabin-core` runtime modules
  - `neuralcabin-node` N-API bridge scaffold
- Added `native/cpp-inference-server/` scaffold.
- Updated user-facing docs:
  - `README.md`
  - `src/renderer/docs.js`
  - `native/rust-engine/README.md`

## Stage 2

- Move autograd graph and optimizer state to Rust.
- Replace JS trainer hot path with N-API engine calls.
- Backfill fused GPU kernels per op family.
- Replace in-app docs TODO markers with concrete native benchmarking guidance.

## Stage 3

- Enable distributed allreduce transports for multi-node runs.
- Finalize ONNX exporter from full compute graph.
- Replace JS API server inference path with C++ service integration.
