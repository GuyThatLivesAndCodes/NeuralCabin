#!/usr/bin/env node
'use strict';

// Microbenchmark for the training hot path.
//
// Measures: time per epoch on a representative charLM config, plus wall
// time for individual ops (matmul, softmax, gelu) at typical shapes. Run
// before/after optimization to get apples-to-apples numbers.
//
// Usage:
//   node scripts/bench-train.js               # default: small charLM
//   node scripts/bench-train.js --big         # bigger config (chat preset)
//   node scripts/bench-train.js --ops         # only the per-op micros

const T = require('../src/engine/tensor');
const { trainNetworkSync } = require('../src/engine/trainer');

function fmt(ms) { return ms.toFixed(2).padStart(8) + ' ms'; }
function fmtRate(n, ms) {
  const per = (ms / n).toFixed(3);
  const rate = (n * 1000 / ms).toFixed(1);
  return `${per} ms/op  (${rate} ops/sec)`;
}

function benchOps() {
  console.log('\n== per-op microbenchmarks ==');
  // Shapes mirror what charLM actually runs:
  //   linear1: matmul [B, L*D] x [L*D, H]
  //   linear2: matmul [B, H] x [H, V]
  // Use B=32 (post-bump default), L=16, D=16, H=64, V=80 — modest config.
  const cases = [
    { name: 'matmul [32,256] x [256,64]', a: [32, 256], b: [256, 64], iters: 200 },
    { name: 'matmul [32,64] x [64,80]',   a: [32, 64],  b: [64, 80],  iters: 500 },
    { name: 'matmul [32,4096] x [4096,96]', a: [32, 4096], b: [4096, 96], iters: 30 }, // chat-preset linear1
    { name: 'matmul [32,96] x [96,96]',   a: [32, 96],  b: [96, 96],  iters: 500 }
  ];
  for (const c of cases) {
    const A = T.randn(c.a, Math.random); A.requiresGrad = true;
    const B = T.randn(c.b, Math.random); B.requiresGrad = true;
    // Warm
    for (let i = 0; i < 5; i++) T.matmul(A, B);
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < c.iters; i++) {
      const out = T.matmul(A, B);
      // Backward as well — that's the dominant cost in real training
      out.grad = new Float32Array(out.size);
      for (let k = 0; k < out.size; k++) out.grad[k] = 1;
      out._backward();
    }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    console.log(`  ${c.name.padEnd(40)} ${fmtRate(c.iters, ms)}`);
  }

  // GELU
  {
    const a = T.randn([32, 96], Math.random); a.requiresGrad = true;
    for (let i = 0; i < 10; i++) T.gelu(a);
    const t0 = process.hrtime.bigint();
    const N = 5000;
    for (let i = 0; i < N; i++) {
      const o = T.gelu(a);
      o.grad = new Float32Array(o.size); for (let k = 0; k < o.size; k++) o.grad[k] = 1;
      o._backward();
    }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    console.log(`  ${'gelu fwd+bwd [32,96]'.padEnd(40)} ${fmtRate(N, ms)}`);
  }

  // Softmax + CE
  {
    const logits = T.randn([32, 80], Math.random); logits.requiresGrad = true;
    const labels = new Array(32).fill(0).map(() => Math.floor(Math.random() * 80));
    for (let i = 0; i < 10; i++) T.softmaxCrossEntropy(logits, labels);
    const t0 = process.hrtime.bigint();
    const N = 5000;
    for (let i = 0; i < N; i++) {
      const o = T.softmaxCrossEntropy(logits, labels);
      o.backward();
    }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    console.log(`  ${'softmaxCE fwd+bwd [32,80]'.padEnd(40)} ${fmtRate(N, ms)}`);
  }
}

function buildBenchNetwork(config, workers) {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz \n.,!?';
  let text = '';
  while (text.length < 5000) text += alphabet;
  return {
    architecture: {
      kind: 'charLM',
      vocabSize: 0,
      embDim: config.embDim,
      contextLen: config.contextLen,
      hidden: config.hidden,
      activation: 'gelu',
      dropout: 0
    },
    training: {
      optimizer: 'adam', learningRate: 1e-3,
      batchSize: config.batchSize, epochs: config.epochs, seed: 42,
      workers: workers || 0
    },
    trainingData: { text },
    state: null,
    tokenizer: null
  };
}

function benchTrain(config) {
  console.log(`\n== full training: ${config.name} ==`);
  const network = buildBenchNetwork(config, 0);
  const text = network.trainingData.text;
  const t0 = process.hrtime.bigint();
  const result = trainNetworkSync(network, {});
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  const stepsPerEpoch = Math.max(1, Math.floor((text.length - config.contextLen) / config.batchSize));
  const totalSteps = config.epochs * stepsPerEpoch;
  console.log(`  total      = ${fmt(ms)}`);
  console.log(`  per epoch  = ${fmt(ms / config.epochs)}`);
  console.log(`  per step   = ${fmt(ms / totalSteps)}  (${(totalSteps * 1000 / ms).toFixed(1)} steps/sec)`);
  console.log(`  steps      = ${totalSteps}  (${stepsPerEpoch}/epoch x ${config.epochs} epochs)`);
  console.log(`  final loss = ${result.metrics[result.metrics.length - 1].loss.toFixed(4)}`);
}

async function benchTrainParallel(config, workers) {
  const { trainNetwork } = require('../src/engine/trainer');
  console.log(`\n== full training (PARALLEL ${workers} workers): ${config.name} ==`);
  const network = buildBenchNetwork(config, workers);
  const text = network.trainingData.text;
  const t0 = process.hrtime.bigint();
  const result = await trainNetwork(network, {});
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  // In parallel mode each "step" processes workers× more data, so steps/epoch
  // is divided accordingly. Effective batch reflects the true gradient batch.
  const stepsPerEpoch = Math.max(1, Math.floor((text.length - config.contextLen) / (config.batchSize * workers)));
  const totalSteps = config.epochs * stepsPerEpoch;
  console.log(`  total      = ${fmt(ms)}`);
  console.log(`  per epoch  = ${fmt(ms / config.epochs)}`);
  console.log(`  per step   = ${fmt(ms / totalSteps)}  (${(totalSteps * 1000 / ms).toFixed(1)} steps/sec)`);
  console.log(`  steps      = ${totalSteps}  (${stepsPerEpoch}/epoch x ${config.epochs} epochs, eff. batch=${config.batchSize * workers})`);
  console.log(`  final loss = ${result.metrics[result.metrics.length - 1].loss.toFixed(4)}`);
}

async function main() {
  const args = process.argv.slice(2);
  const wantOps = args.includes('--ops') || args.includes('--all');
  const wantBig = args.includes('--big') || args.includes('--all');
  const wantParallel = args.includes('--parallel') || args.includes('--all');
  const wantSmall = !wantOps || args.includes('--small') || args.includes('--all') || (!wantBig && !wantOps && !wantParallel);

  const small = {
    name: 'standard charLM (B=32, L=16, D=16, H=[64], 5 epochs)',
    embDim: 16, contextLen: 16, hidden: [64], batchSize: 32, epochs: 5
  };
  const big = {
    name: 'chat preset (B=32, L=128, D=32, H=[96,96], 3 epochs)',
    embDim: 32, contextLen: 128, hidden: [96, 96], batchSize: 32, epochs: 3
  };

  if (wantOps) benchOps();
  if (wantSmall) benchTrain(small);
  if (wantBig) benchTrain(big);
  if (wantParallel) {
    const cpus = require('os').cpus().length;
    const workers = Math.min(8, Math.max(2, cpus - 1));
    await benchTrainParallel(big, workers);
    await benchTrainParallel(small, workers);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
