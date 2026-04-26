# NeuralCabin Rust Engine

This workspace is the staged replacement for the legacy JavaScript tensor engine.

## Crates

- `neuralcabin-core`: Core tensor runtime with CPU SIMD + rayon fallback, WebGPU runtime scaffolding, mixed precision dtypes, mmap streaming, fusion planner, SPIR-V JIT entry points, distributed allreduce contracts, and ONNX AOT export hooks.
- `neuralcabin-node`: Node/N-API bridge used by `src/engine/tensor-native-loader.js`.

## Build (optional)

```bash
cargo check --manifest-path native/rust-engine/Cargo.toml -p neuralcabin-core
cargo build --manifest-path native/rust-engine/Cargo.toml -p neuralcabin-node --release
```

The JS app auto-detects the native module when available and falls back to the JS backend otherwise.

## Runtime selection from JS

- `NEURALCABIN_ENGINE_BACKEND=auto` (default)
- `NEURALCABIN_ENGINE_BACKEND=js`
- `NEURALCABIN_ENGINE_BACKEND=rust`
- `NEURALCABIN_NATIVE_BINDING=/absolute/path/to/native.node`
