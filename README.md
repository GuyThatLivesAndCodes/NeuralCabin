# NeuralCity

Build neural networks from the ground up. No NumPy, no TensorFlow, no PyTorch — every tensor op, every gradient, every optimizer step is plain JavaScript code you can read and modify.

## What's inside

- **Custom engine** (`src/engine/`) — Tensor + autograd, layers (Linear, Activation, Dropout, Embedding), optimizers (SGD, Adam), char-level tokenizer, classifier / regressor / character LM trainers.
- **NeuralScript DSL** (`src/dsl/`) — A tiny `do/end`-block language with first-class neural primitives (`build`, `train`, `predict`).
- **Electron app** (`src/main/`, `src/renderer/`) — Black & white UI with Networks list, Editor, Train, Inference, API, Script, and Docs tabs.
- **In-app HTTP API server** — Serve any trained model on a local port. Other devices on your network can `POST /predict`.
- **Encryption at rest** — Optional per-network AES-256-GCM with scrypt-derived key.
- **Templates** — XOR, 2D Spiral, Sine Regressor, Tiny Char LM, Code Predictor.

## Run from source

```bash
npm install
npm start          # launches the app
npm test           # runs the engine test harness (20 tests)
npm run build:win  # produces dist/NeuralCity-Setup-1.0.0.exe
```

## Install on Windows

After building, run `dist/NeuralCity-Setup-1.0.0.exe`. The NSIS installer lets you pick an install location, creates a Start Menu and desktop shortcut, and registers an uninstaller.

## Layout

```
src/
  engine/      tensor.js, layers.js, optim.js, model.js, tokenizer.js, trainer.js
  dsl/         lexer.js, parser.js, interpreter.js
  main/        main.js, preload.js, storage.js, training-manager.js, api-server.js
  renderer/    index.html, styles.css, app.js, templates.js, docs.js
tests/         run-tests.js
assets/        icon.png, make-icon.js
```

## Philosophy

The whole engine is roughly 2,000 lines of plain JavaScript. Open `src/engine/tensor.js`, read the `matmul` function, and verify every multiplication. Add a new activation. Try a different optimizer. The tools are yours.
