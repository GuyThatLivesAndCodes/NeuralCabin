'use strict';

const path = require('path');

function loadNativeBinding() {
  const attempts = [
    './index.node',
    './neuralcabin-node.node',
    './neuralcabin_node.node',
    './neuralcabin-node.win32-x64-msvc.node',
    './neuralcabin_node.win32-x64-msvc.node',
    './neuralcabin-node.linux-x64-gnu.node',
    './neuralcabin_node.linux-x64-gnu.node',
    './neuralcabin-node.darwin-x64.node',
    './neuralcabin_node.darwin-x64.node',
  ];
  let lastErr = null;
  for (const entry of attempts) {
    try { return require(entry); } catch (err) { lastErr = err; }
  }
  if (lastErr) throw lastErr;
  throw new Error('No native binding candidate could be loaded.');
}

const native = loadNativeBinding();
const fallback = require(path.resolve(__dirname, '..', '..', '..', 'src', 'engine', 'tensor-js'));

// ── helpers ───────────────────────────────────────────────────────────────────

function toF64(v) {
  if (v instanceof Float32Array || v instanceof Float64Array) return Array.from(v);
  if (Array.isArray(v)) return v;
  return v;
}

function toShape32(s) { return s.map(x => x >>> 0); }

// ── Build the complete drop-in API using Rust for all compute-heavy ops ───────

const api = { ...fallback };

// matmul: a[B,K] @ b[K,N] → out[B,N]
api.matmul = function rustMatmul(a, b) {
  if (!a || !b || !a.shape || !b.shape || a.shape.length !== 2 || b.shape.length !== 2) {
    return fallback.matmul(a, b);
  }
  const [m, k] = a.shape;
  const [k2, n] = b.shape;
  if (k !== k2) return fallback.matmul(a, b);

  const out = new fallback.Tensor([m, n], null, a.requiresGrad || b.requiresGrad);
  const result = native.matmul(toF64(a.data), m, k, toF64(b.data), k2, n);
  out.data = new Float32Array(result);
  out._parents = [a, b];
  out._backward = () => {
    if (a.requiresGrad) {
      a.ensureGrad();
      const G = out.grad, A = a.data, Bd = b.data, dA = a.grad;
      for (let i = 0; i < m; i++) {
        const rowG = i * n, rowDA = i * k;
        for (let kk = 0; kk < k; kk++) {
          const rowB = kk * n; let s = 0;
          for (let j = 0; j < n; j++) s += G[rowG + j] * Bd[rowB + j];
          dA[rowDA + kk] += s;
        }
      }
    }
    if (b.requiresGrad) {
      b.ensureGrad();
      const G = out.grad, A = a.data, dB = b.grad;
      for (let i = 0; i < m; i++) {
        const rowA = i * k, rowG = i * n;
        for (let kk = 0; kk < k; kk++) {
          const aik = A[rowA + kk], rowB = kk * n;
          for (let j = 0; j < n; j++) dB[rowB + j] += aik * G[rowG + j];
        }
      }
    }
  };
  return out;
};

// add: same-shape or bias-broadcast [B,N] + [N]
api.add = function rustAdd(a, b) {
  const sameShape = fallback.sameShape(a.shape, b.shape);
  const isBiasBroadcast = a.shape.length === 2 && b.shape.length === 1 && a.shape[1] === b.shape[0];
  if (!sameShape && !isBiasBroadcast) return fallback.add(a, b);

  const [arows, acols] = a.shape.length === 2 ? a.shape : [1, a.shape[0]];
  const [brows, bcols] = b.shape.length === 2 ? b.shape : [1, b.shape[0]];
  const result = native.addOp(toF64(a.data), arows, acols, toF64(b.data), brows, bcols);
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad || b.requiresGrad);
  out._parents = [a, b];
  out._backward = () => {
    const G = out.grad;
    if (a.requiresGrad) { a.ensureGrad(); const dA = a.grad; for (let i = 0; i < a.size; i++) dA[i] += G[i]; }
    if (b.requiresGrad) {
      b.ensureGrad(); const dB = b.grad;
      if (sameShape) { for (let i = 0; i < b.size; i++) dB[i] += G[i]; }
      else { const [B, N] = a.shape; for (let i = 0; i < B; i++) { const row = i * N; for (let j = 0; j < N; j++) dB[j] += G[row + j]; } }
    }
  };
  return out;
};

// sub
api.sub = function rustSub(a, b) {
  if (!fallback.sameShape(a.shape, b.shape)) return fallback.sub(a, b);
  const [rows, cols] = a.shape.length === 2 ? a.shape : [1, a.size];
  const result = native.subOp(toF64(a.data), toF64(b.data), rows, cols);
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad || b.requiresGrad);
  out._parents = [a, b];
  out._backward = () => {
    if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i]; }
    if (b.requiresGrad) { b.ensureGrad(); for (let i = 0; i < b.size; i++) b.grad[i] -= out.grad[i]; }
  };
  return out;
};

// mul
api.mul = function rustMul(a, b) {
  if (typeof b === 'number') return fallback.mul(a, b);
  if (!fallback.sameShape(a.shape, b.shape)) return fallback.mul(a, b);
  const [rows, cols] = a.shape.length === 2 ? a.shape : [1, a.size];
  const result = native.mulOp(toF64(a.data), toF64(b.data), rows, cols);
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad || b.requiresGrad);
  out._parents = [a, b];
  out._backward = () => {
    const G = out.grad, Ad = a.data, Bd = b.data;
    if (a.requiresGrad) { a.ensureGrad(); const dA = a.grad; for (let i = 0; i < a.size; i++) dA[i] += G[i] * Bd[i]; }
    if (b.requiresGrad) { b.ensureGrad(); const dB = b.grad; for (let i = 0; i < b.size; i++) dB[i] += G[i] * Ad[i]; }
  };
  return out;
};

// relu
api.relu = function rustRelu(a) {
  const result = native.reluOp(toF64(a.data), toShape32(a.shape));
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, O = out.data, dA = a.grad;
    for (let i = 0; i < a.size; i++) if (O[i] > 0) dA[i] += G[i];
  };
  return out;
};

// leakyRelu
api.leakyRelu = function rustLeakyRelu(a, alpha = 0.01) {
  const result = native.leakyReluOp(toF64(a.data), toShape32(a.shape), alpha);
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, A = a.data, dA = a.grad;
    for (let i = 0; i < a.size; i++) dA[i] += G[i] * (A[i] > 0 ? 1 : alpha);
  };
  return out;
};

// tanh
api.tanh = function rustTanh(a) {
  const result = native.tanhOp(toF64(a.data), toShape32(a.shape));
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, O = out.data, dA = a.grad;
    for (let i = 0; i < a.size; i++) { const t = O[i]; dA[i] += G[i] * (1 - t * t); }
  };
  return out;
};

// sigmoid
api.sigmoid = function rustSigmoid(a) {
  const result = native.sigmoidOp(toF64(a.data), toShape32(a.shape));
  const out = new fallback.Tensor(a.shape, new Float32Array(result), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, O = out.data, dA = a.grad;
    for (let i = 0; i < a.size; i++) { const s = O[i]; dA[i] += G[i] * s * (1 - s); }
  };
  return out;
};

// gelu — Rust also returns tcache so backward avoids a re-compute
api.gelu = function rustGelu(a) {
  const { output, tcache } = native.geluOp(toF64(a.data), toShape32(a.shape));
  const out = new fallback.Tensor(a.shape, new Float32Array(output), a.requiresGrad);
  const c = Math.sqrt(2 / Math.PI);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, A = a.data, dA = a.grad;
    for (let i = 0; i < a.size; i++) {
      const x = A[i], t = tcache[i];
      const dt = (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
      dA[i] += G[i] * (0.5 * (1 + t) + 0.5 * x * dt);
    }
  };
  return out;
};

// softmax
api.softmax = function rustSoftmax(a) {
  if (a.shape.length !== 2) return fallback.softmax(a);
  const [B, N] = a.shape;
  const result = native.softmaxOp(toF64(a.data), B, N);
  const out = new fallback.Tensor([B, N], new Float32Array(result), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, O = out.data, dA = a.grad;
    for (let i = 0; i < B; i++) {
      const row = i * N; let dot = 0;
      for (let j = 0; j < N; j++) dot += G[row + j] * O[row + j];
      for (let j = 0; j < N; j++) dA[row + j] += O[row + j] * (G[row + j] - dot);
    }
  };
  return out;
};

// softmaxCrossEntropy — Rust returns probs so backward is O(B*C) without re-compute
api.softmaxCrossEntropy = function rustSCE(logits, labels) {
  if (logits.shape.length !== 2) return fallback.softmaxCrossEntropy(logits, labels);
  const [B, C] = logits.shape;
  const { loss, probs } = native.softmaxCrossEntropyOp(toF64(logits.data), labels, B, C);
  const out = new fallback.Tensor([1], new Float32Array([loss]), logits.requiresGrad);
  out._parents = [logits];
  out._backward = () => {
    if (!logits.requiresGrad) return;
    logits.ensureGrad();
    const g = out.grad[0] / B, dL = logits.grad;
    for (let i = 0; i < B; i++) {
      const row = i * C;
      for (let j = 0; j < C; j++) dL[row + j] += g * probs[row + j];
      dL[row + labels[i]] -= g;
    }
  };
  return out;
};

// mseLoss
api.mseLoss = function rustMseLoss(a, b) {
  if (!fallback.sameShape(a.shape, b.shape)) return fallback.mseLoss(a, b);
  const sh = toShape32(a.shape);
  const lossVal = native.mseLossOp(toF64(a.data), toF64(b.data), sh);
  const out = new fallback.Tensor([1], new Float32Array([lossVal]), a.requiresGrad || b.requiresGrad);
  out._parents = [a, b];
  out._backward = () => {
    const scale = (2 / a.size) * out.grad[0];
    if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += scale * (a.data[i] - b.data[i]); }
    if (b.requiresGrad) { b.ensureGrad(); for (let i = 0; i < b.size; i++) b.grad[i] -= scale * (a.data[i] - b.data[i]); }
  };
  return out;
};

// dropout — Rust generates the mask
api.dropout = function rustDropout(a, p, training, rng) {
  if (!training || p <= 0) return fallback.dropout(a, p, training, rng);
  // Use a deterministic seed derived from the JS rng so reproducibility is maintained.
  const seed = rng ? Math.floor(rng() * 0xFFFFFFFF) : (Math.random() * 0xFFFFFFFF) | 0;
  const { output, mask } = native.dropoutOp(toF64(a.data), toShape32(a.shape), p, seed >>> 0);
  const maskF32 = new Float32Array(mask);
  const out = new fallback.Tensor(a.shape, new Float32Array(output), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * maskF32[i];
  };
  return out;
};

// embedding
api.embedding = function rustEmbedding(weights, ids) {
  if (weights.shape.length !== 2) return fallback.embedding(weights, ids);
  const [V, D] = weights.shape;
  const result = native.embeddingOp(toF64(weights.data), V, D, ids);
  const B = ids.length;
  const out = new fallback.Tensor([B, D], new Float32Array(result), weights.requiresGrad);
  out._parents = [weights];
  out._backward = () => {
    if (!weights.requiresGrad) return;
    weights.ensureGrad();
    const gradW = native.embeddingBackwardOp(toF64(out.grad), ids, V, D);
    const dW = weights.grad;
    for (let i = 0; i < dW.length; i++) dW[i] += gradW[i];
  };
  return out;
};

// sumAll / mean — delegate sumAll to Rust, mean reuses it
api.sumAll = function rustSumAll(a) {
  const sh = toShape32(a.shape);
  const s = native.sumAllOp(toF64(a.data), sh);
  const out = new fallback.Tensor([1], new Float32Array([s]), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[0];
  };
  return out;
};

api.mean = function rustMean(a) {
  const s = api.sumAll(a);
  return api.mul(s, 1 / a.size);
};

// randn — Rust Box-Muller with Mulberry32 (matches JS seed behaviour)
api.randn = function rustRandn(shape, rng) {
  const seed = rng ? Math.floor(rng() * 0xFFFFFFFF) : (Math.random() * 0xFFFFFFFF) | 0;
  const result = native.randnOp(toShape32(shape), seed >>> 0);
  return new fallback.Tensor(shape, new Float32Array(result), false);
};

// ── Extra surface area exposed to JS consumers ────────────────────────────────

api.backendInfo = () => native.backendInfo();
api.initBackend = (mode, precision) => native.initBackend(mode, precision);
api.hasNanOrInf = (t) => native.hasNanOrInfOp(toF64(t.data), toShape32(t.shape));

// Optimizer helpers (pure-math Rust, state lives on JS side)
api.rust = {
  sgdStep: (params, grads, buf, lr, momentum, wd) =>
    native.sgdStep(params, grads, buf, lr, momentum, wd),
  adamStep: (params, grads, m, v, lr, b1, b2, eps, t) =>
    native.adamStep(params, grads, m, v, lr, b1, b2, eps, t),
  adamwStep: (params, grads, m, v, lr, b1, b2, eps, wd, t) =>
    native.adamwStep(params, grads, m, v, lr, b1, b2, eps, wd, t),
  clipGradNorm: (grads, maxNorm) => native.clipGradNorm(grads, maxNorm),

  // Q-Learning
  epsilonGreedy: (qValues, epsilon, seed) => native.epsilonGreedy(qValues, epsilon, seed),
  computeTdTargets: (rewards, nextQ, dones, gamma, nActions) =>
    native.computeTdTargets(rewards, nextQ, dones, gamma, nActions),
  dqnLoss: (qValues, actions, targets, nActions) =>
    native.dqnLoss(qValues, actions, targets, nActions),
  dqnHuberLoss: (qValues, actions, targets, nActions, delta) =>
    native.dqnHuberLoss(qValues, actions, targets, nActions, delta),
  softUpdateTarget: (target, online, tau) => native.softUpdateTarget(target, online, tau),
  replayBufferSample: (states, actions, rewards, nextStates, dones, size, sDim, aDim, batchSize, seed) =>
    native.replayBufferSample(states, actions, rewards, nextStates, dones, size, sDim, aDim, batchSize, seed),

  // Neuroevolution
  neMutate: (params, pMutate, std, seed) => native.neMutate(params, pMutate, std, seed),
  neCrossoverUniform: (p1, p2, seed) => native.neCrossoverUniform(p1, p2, seed),
  neCrossoverArithmetic: (p1, p2, alpha) => native.neCrossoverArithmetic(p1, p2, alpha),
  neTournamentSelect: (fitnesses, k, seed) => native.neTournamentSelect(fitnesses, k, seed),
  neTruncationSelect: (fitnesses, n) => native.neTruncationSelect(fitnesses, n),
  neEvolveGeneration: (flatParams, fitnesses, paramCount, eliteCount, pMutate, std, k, seed) =>
    native.neEvolveGeneration(flatParams, fitnesses, paramCount, eliteCount, pMutate, std, k, seed),
  neFitnessStats: (fitnesses) => native.neFitnessStats(fitnesses),

  // Layer helpers
  linearForward: (input, b, inF, weight, outF, bias) =>
    native.linearForward(input, b, inF, weight, outF, bias),
};

module.exports = { version: '0.2.0', api };
