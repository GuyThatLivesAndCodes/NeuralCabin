# NeuralCity

Build neural networks from the ground up. No NumPy, no TensorFlow, no PyTorch — every tensor op, every gradient, every optimizer step is plain JavaScript code you can read and modify.

## What's inside

- **Custom engine** (`src/engine/`) — Tensor + autograd, layers (Linear, Activation, Dropout, Embedding), optimizers (SGD, Adam) with persisted momentum/variance state, char-level tokenizer with append-only vocab extension, classifier / regressor / character LM trainers.
- **Multi-turn chat** — Train on `user → assistant → user → assistant` conversations and chat with the model in-app. Multi-turn history is also exposed via the HTTP API (stateless `/predict` or stateful `/chat`).
- **NeuralScript DSL** (`src/dsl/`) — A tiny `do/end`-block language with first-class neural primitives (`build`, `train`, `predict`).
- **Electron app** (`src/main/`, `src/renderer/`) — Black & white UI with Networks list, Editor, Train, Inference / Chat, API, Script, and Docs tabs.
- **In-app HTTP API server** — Serve any trained model on a local port. Other devices on your network can `POST /predict` or, for chat models, `POST /chat`.
- **Encryption at rest** — Optional per-network AES-256-GCM with scrypt-derived key.
- **Templates** — XOR, 2D Spiral, Sine Regressor, Tiny Char LM, Code Predictor.

## Run from source

```bash
npm install
npm start          # launches the app
npm test           # runs the engine test harness
npm run build:win  # produces dist/NeuralCity-Setup-1.0.0.exe
```

## Install on Windows

After building, run `dist/NeuralCity-Setup-1.0.0.exe`. The NSIS installer lets you pick an install location, creates a Start Menu and desktop shortcut, and registers an uninstaller.

Releases are published on GitHub by the project maintainer.

## Layout

```
src/
  engine/      tensor.js, layers.js, optim.js, model.js, tokenizer.js, chat-format.js, trainer.js
  dsl/         lexer.js, parser.js, interpreter.js
  main/        main.js, preload.js, storage.js, training-manager.js, api-server.js
  renderer/    index.html, styles.css, app.js, templates.js, docs.js
tests/         run-tests.js
assets/        icon.png, make-icon.js
```

## Multi-turn chat at a glance

Training data — mix in samples with several alternating turns so the model learns to *continue* a thread, not just answer one prompt:

```json
{ "samples": [
  { "messages": [
      { "role": "system",    "content": "You are concise." },
      { "role": "user",      "content": "Hi" },
      { "role": "assistant", "content": "Hello!" },
      { "role": "user",      "content": "How are you?" },
      { "role": "assistant", "content": "I am well, thanks." }
  ] }
] }
```

HTTP — stateless: you pass the running history each call.

```bash
curl -X POST http://localhost:PORT/predict -H "Content-Type: application/json" -d '{
  "history":[
    {"role":"user","content":"Hi"},
    {"role":"assistant","content":"Hello!"}
  ],
  "prompt":"How are you?",
  "system":"Be concise."
}'
```

HTTP — stateful: the server keeps the thread keyed by `sessionId`.

```bash
curl -X POST http://localhost:PORT/chat -H "Content-Type: application/json" \
  -d '{"message":"Hi","system":"Be concise."}'
# → {"sessionId":"session-…","reply":"…","history":[…]}
curl -X POST http://localhost:PORT/chat -H "Content-Type: application/json" \
  -d '{"sessionId":"session-…","message":"How are you?"}'
```

## Philosophy

The whole engine is roughly 2,000 lines of plain JavaScript. Open `src/engine/tensor.js`, read the `matmul` function, and verify every multiplication. Add a new activation. Try a different optimizer. The tools are yours.

## Contributing

The project will be open-sourced on GitHub. Issues and pull requests are welcome.
