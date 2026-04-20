'use strict';

// Worker process for data-parallel training. The main thread spawns one of
// these per CPU it wants to use; each worker holds a full copy of the model
// and processes one batch per step.
//
// Memory model — weights live in SharedArrayBuffers handed in via workerData,
// so all workers (and the main thread) read the SAME bytes for forward
// passes. Gradients are written to per-worker SharedArrayBuffer slots, then
// the main thread sums them and applies the optimizer step. Weight updates
// land in the shared buffer in-place, so the next forward pass picks them up
// without any IPC round-trip.
//
// Per-step protocol (over postMessage):
//   main → worker:  { type: 'step', batch: int[][], labels: int[] }
//   worker → main:  { type: 'done', loss: number }   // grads are already in SAB
//
// Setup happens once via workerData; per-step messages carry only batch ids
// (small — a few KB at most), keeping the round-trip latency minimal.

const { parentPort, workerData } = require('worker_threads');
const T = require('./tensor');
const { buildFromSpec, restoreFromState } = require('./model');

const { archSpec, modelState, paramSpecs, gradOffsets, weightSAB, gradSAB } = workerData;

// Build a fresh model with the architecture, then *swap in* the SAB-backed
// Float32Array views for every parameter's data and grad. Both .data and .grad
// will then alias the shared memory exactly — no copies.
const rng = T.rngFromSeed((archSpec.seed | 0) || 42);
const model = modelState
  ? restoreFromState(modelState, archSpec, rng)
  : buildFromSpec(archSpec, rng);

const weightView = new Float32Array(weightSAB);
const gradView = new Float32Array(gradSAB);

// Walk model params in the same order the main thread used (model.params).
// Wire each param's .data to a slice of the shared weight buffer, and pre-
// install a .grad alias to a slice of the per-worker grad buffer.
for (let i = 0; i < model.params.length; i++) {
  const p = model.params[i];
  const spec = paramSpecs[i];
  if (p.size !== spec.size) {
    throw new Error(`worker: param ${i} size mismatch (model=${p.size} spec=${spec.size})`);
  }
  // Replace the param's data with a view into shared memory. After this, any
  // read of p.data[j] sees the latest value the main thread wrote.
  p.data = weightView.subarray(spec.offset, spec.offset + spec.size);
  // Pre-install grad slot; backward() will accumulate into this view.
  p.grad = gradView.subarray(gradOffsets[i], gradOffsets[i] + spec.size);
}

function runStep(msg) {
  // Zero the grad slots for this worker (only this worker writes into them).
  gradView.fill(0, gradOffsets[0], gradOffsets[gradOffsets.length - 1] + paramSpecs[paramSpecs.length - 1].size);

  let logits;
  if (archSpec.kind === 'charLM') {
    logits = model.forward(msg.batch, { training: true, rng });
  } else if (archSpec.kind === 'regressor') {
    const xb = new T.Tensor([msg.batchSize, archSpec.inputDim], new Float32Array(msg.batchData), false);
    logits = model.forward(xb, { training: true, rng });
  } else {
    const xb = new T.Tensor([msg.batchSize, archSpec.inputDim], new Float32Array(msg.batchData), false);
    logits = model.forward(xb, { training: true, rng });
  }

  let loss;
  if (archSpec.kind === 'regressor') {
    const yb = new T.Tensor([msg.batchSize, archSpec.outputDim], new Float32Array(msg.labelData), false);
    loss = T.mseLoss(logits, yb);
  } else {
    loss = T.softmaxCrossEntropy(logits, msg.labels);
  }
  loss.backward();
  parentPort.postMessage({ type: 'done', loss: loss.data[0] });
}

parentPort.on('message', (msg) => {
  try {
    if (msg.type === 'step') return runStep(msg);
    if (msg.type === 'shutdown') process.exit(0);
  } catch (e) {
    parentPort.postMessage({ type: 'error', message: e.message, stack: e.stack });
  }
});
