// NeuralCity tensor + autograd engine. Pure JS, no dependencies.
// Tensors store a flat Float32Array + shape. Autograd builds a dynamic graph
// on forward, then walks it in reverse topological order on .backward().

'use strict';

function shapeSize(shape) {
  let s = 1;
  for (let i = 0; i < shape.length; i++) s *= shape[i];
  return s;
}

function sameShape(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

let _id = 0;

class Tensor {
  // shape: number[]; data: Float32Array | number[]
  constructor(shape, data, requiresGrad = false) {
    this.id = ++_id;
    this.shape = shape.slice();
    this.size = shapeSize(shape);
    if (data == null) {
      this.data = new Float32Array(this.size);
    } else if (data instanceof Float32Array) {
      if (data.length !== this.size) throw new Error(`data length ${data.length} != size ${this.size}`);
      this.data = data;
    } else {
      if (data.length !== this.size) throw new Error(`data length ${data.length} != size ${this.size}`);
      this.data = new Float32Array(data);
    }
    this.requiresGrad = requiresGrad;
    this.grad = null; // Float32Array when populated
    this._backward = null; // function() called during backward
    this._parents = []; // Tensor[]
  }

  zeroGrad() {
    if (this.grad) this.grad.fill(0);
    else this.grad = new Float32Array(this.size);
  }

  ensureGrad() {
    if (!this.grad) this.grad = new Float32Array(this.size);
    return this.grad;
  }

  detach() {
    const t = new Tensor(this.shape, new Float32Array(this.data), false);
    return t;
  }

  toJSON() {
    return { shape: this.shape, data: Array.from(this.data) };
  }

  static fromJSON(obj, requiresGrad = false) {
    return new Tensor(obj.shape, new Float32Array(obj.data), requiresGrad);
  }

  // Walk graph in reverse topo order and accumulate grads.
  backward(seedGrad) {
    const topo = [];
    const seen = new Set();
    (function build(t) {
      if (seen.has(t.id)) return;
      seen.add(t.id);
      for (const p of t._parents) build(p);
      topo.push(t);
    })(this);

    // Seed
    if (!this.grad) this.grad = new Float32Array(this.size);
    if (seedGrad) {
      for (let i = 0; i < this.size; i++) this.grad[i] += seedGrad[i];
    } else {
      if (this.size !== 1) throw new Error('backward() on non-scalar requires seedGrad');
      this.grad[0] += 1;
    }

    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t._backward) t._backward();
    }
  }
}

// ---------- constructors ----------

function tensor(shape, data, requiresGrad = false) {
  return new Tensor(shape, data, requiresGrad);
}

function zeros(shape, requiresGrad = false) {
  return new Tensor(shape, null, requiresGrad);
}

function ones(shape, requiresGrad = false) {
  const t = new Tensor(shape, null, requiresGrad);
  t.data.fill(1);
  return t;
}

// Mulberry32 for reproducible init
function rngFromSeed(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(shape, rng) {
  const size = shapeSize(shape);
  const data = new Float32Array(size);
  const r = rng || Math.random;
  for (let i = 0; i < size; i += 2) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    data[i] = mag * Math.cos(2.0 * Math.PI * v);
    if (i + 1 < size) data[i + 1] = mag * Math.sin(2.0 * Math.PI * v);
  }
  return new Tensor(shape, data, false);
}

// ---------- helpers ----------

function assert2D(t, name) {
  if (t.shape.length !== 2) throw new Error(`${name} must be 2D`);
}

// ---------- ops (all build graph when requiresGrad) ----------

function add(a, b) {
  // broadcast: b may be shape [N] with a shape [B,N] (bias)
  let out;
  if (sameShape(a.shape, b.shape)) {
    out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
    for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] + b.data[i];
  } else if (a.shape.length === 2 && b.shape.length === 1 && a.shape[1] === b.shape[0]) {
    const [B, N] = a.shape;
    out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < N; j++) {
        out.data[i * N + j] = a.data[i * N + j] + b.data[j];
      }
    }
  } else {
    throw new Error(`add: incompatible shapes ${a.shape} and ${b.shape}`);
  }
  out._parents = [a, b];
  out._backward = () => {
    if (a.requiresGrad) {
      a.ensureGrad();
      for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i];
    }
    if (b.requiresGrad) {
      b.ensureGrad();
      if (sameShape(a.shape, b.shape)) {
        for (let i = 0; i < b.size; i++) b.grad[i] += out.grad[i];
      } else {
        const [B, N] = a.shape;
        for (let i = 0; i < B; i++) {
          for (let j = 0; j < N; j++) b.grad[j] += out.grad[i * N + j];
        }
      }
    }
  };
  return out;
}

function sub(a, b) {
  if (!sameShape(a.shape, b.shape)) throw new Error('sub: shape mismatch');
  const out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] - b.data[i];
  out._parents = [a, b];
  out._backward = () => {
    if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i]; }
    if (b.requiresGrad) { b.ensureGrad(); for (let i = 0; i < b.size; i++) b.grad[i] -= out.grad[i]; }
  };
  return out;
}

function mul(a, b) {
  // elementwise, same shape or scalar
  let out;
  if (typeof b === 'number') {
    out = new Tensor(a.shape, null, a.requiresGrad);
    for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * b;
    out._parents = [a];
    out._backward = () => {
      if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * b; }
    };
    return out;
  }
  if (!sameShape(a.shape, b.shape)) throw new Error('mul: shape mismatch');
  out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * b.data[i];
  out._parents = [a, b];
  out._backward = () => {
    if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * b.data[i]; }
    if (b.requiresGrad) { b.ensureGrad(); for (let i = 0; i < b.size; i++) b.grad[i] += out.grad[i] * a.data[i]; }
  };
  return out;
}

// Matrix multiply: a[B,K] * b[K,N] => out[B,N]
function matmul(a, b) {
  assert2D(a, 'matmul a'); assert2D(b, 'matmul b');
  const [B, K] = a.shape;
  const [K2, N] = b.shape;
  if (K !== K2) throw new Error(`matmul shape mismatch ${a.shape} x ${b.shape}`);
  const out = new Tensor([B, N], null, a.requiresGrad || b.requiresGrad);
  const A = a.data, Bd = b.data, O = out.data;
  for (let i = 0; i < B; i++) {
    for (let k = 0; k < K; k++) {
      const aik = A[i * K + k];
      if (aik === 0) continue;
      const rowB = k * N;
      const rowO = i * N;
      for (let j = 0; j < N; j++) O[rowO + j] += aik * Bd[rowB + j];
    }
  }
  out._parents = [a, b];
  out._backward = () => {
    const G = out.grad;
    if (a.requiresGrad) {
      a.ensureGrad();
      // dA[i,k] += sum_j G[i,j] * B[k,j]
      for (let i = 0; i < B; i++) {
        for (let j = 0; j < N; j++) {
          const g = G[i * N + j];
          if (g === 0) continue;
          for (let k = 0; k < K; k++) a.grad[i * K + k] += g * Bd[k * N + j];
        }
      }
    }
    if (b.requiresGrad) {
      b.ensureGrad();
      // dB[k,j] += sum_i A[i,k] * G[i,j]
      for (let i = 0; i < B; i++) {
        for (let k = 0; k < K; k++) {
          const aik = A[i * K + k];
          if (aik === 0) continue;
          for (let j = 0; j < N; j++) b.grad[k * N + j] += aik * G[i * N + j];
        }
      }
    }
  };
  return out;
}

function relu(a) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] > 0 ? a.data[i] : 0;
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) if (a.data[i] > 0) a.grad[i] += out.grad[i];
  };
  return out;
}

function leakyRelu(a, alpha = 0.01) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] > 0 ? a.data[i] : a.data[i] * alpha;
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * (a.data[i] > 0 ? 1 : alpha);
  };
  return out;
}

function tanh(a) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  for (let i = 0; i < a.size; i++) out.data[i] = Math.tanh(a.data[i]);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * (1 - out.data[i] * out.data[i]);
  };
  return out;
}

function sigmoid(a) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  for (let i = 0; i < a.size; i++) {
    const x = a.data[i];
    out.data[i] = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) {
      const s = out.data[i];
      a.grad[i] += out.grad[i] * s * (1 - s);
    }
  };
  return out;
}

function gelu(a) {
  // GELU approx
  const out = new Tensor(a.shape, null, a.requiresGrad);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < a.size; i++) {
    const x = a.data[i];
    out.data[i] = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) {
      const x = a.data[i];
      const t = Math.tanh(c * (x + 0.044715 * x * x * x));
      const dt = (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
      const d = 0.5 * (1 + t) + 0.5 * x * dt;
      a.grad[i] += out.grad[i] * d;
    }
  };
  return out;
}

// Softmax along last axis of 2D: shape [B, N]
function softmax(a) {
  assert2D(a, 'softmax');
  const [B, N] = a.shape;
  const out = new Tensor([B, N], null, a.requiresGrad);
  for (let i = 0; i < B; i++) {
    let maxv = -Infinity;
    for (let j = 0; j < N; j++) if (a.data[i * N + j] > maxv) maxv = a.data[i * N + j];
    let sum = 0;
    for (let j = 0; j < N; j++) { const e = Math.exp(a.data[i * N + j] - maxv); out.data[i * N + j] = e; sum += e; }
    for (let j = 0; j < N; j++) out.data[i * N + j] /= sum;
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < B; i++) {
      let dot = 0;
      for (let j = 0; j < N; j++) dot += out.grad[i * N + j] * out.data[i * N + j];
      for (let j = 0; j < N; j++) {
        const s = out.data[i * N + j];
        a.grad[i * N + j] += s * (out.grad[i * N + j] - dot);
      }
    }
  };
  return out;
}

// Cross entropy for softmax-ed probs vs integer labels.
// logits: [B, C]; labels: int array length B. Returns scalar loss.
function softmaxCrossEntropy(logits, labels) {
  assert2D(logits, 'softmaxCrossEntropy');
  const [B, C] = logits.shape;
  const probs = new Float32Array(B * C);
  let loss = 0;
  for (let i = 0; i < B; i++) {
    let maxv = -Infinity;
    for (let j = 0; j < C; j++) if (logits.data[i * C + j] > maxv) maxv = logits.data[i * C + j];
    let sum = 0;
    for (let j = 0; j < C; j++) { const e = Math.exp(logits.data[i * C + j] - maxv); probs[i * C + j] = e; sum += e; }
    for (let j = 0; j < C; j++) probs[i * C + j] /= sum;
    const y = labels[i];
    if (y < 0 || y >= C) throw new Error(`label ${y} out of range [0,${C})`);
    loss += -Math.log(Math.max(probs[i * C + y], 1e-12));
  }
  loss /= B;
  const out = new Tensor([1], new Float32Array([loss]), logits.requiresGrad);
  out._parents = [logits];
  out._backward = () => {
    if (!logits.requiresGrad) return;
    logits.ensureGrad();
    const g = out.grad[0] / B;
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < C; j++) {
        logits.grad[i * C + j] += g * (probs[i * C + j] - (labels[i] === j ? 1 : 0));
      }
    }
  };
  return out;
}

// MSE loss: mean((a-b)^2)
function mseLoss(a, b) {
  if (!sameShape(a.shape, b.shape)) throw new Error('mse shape mismatch');
  let sum = 0;
  for (let i = 0; i < a.size; i++) { const d = a.data[i] - b.data[i]; sum += d * d; }
  sum /= a.size;
  const out = new Tensor([1], new Float32Array([sum]), a.requiresGrad || b.requiresGrad);
  out._parents = [a, b];
  out._backward = () => {
    const scale = (2 / a.size) * out.grad[0];
    if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += scale * (a.data[i] - b.data[i]); }
    if (b.requiresGrad) { b.ensureGrad(); for (let i = 0; i < b.size; i++) b.grad[i] -= scale * (a.data[i] - b.data[i]); }
  };
  return out;
}

// Dropout (training only). Zeros elements with probability p, scales by 1/(1-p).
function dropout(a, p, training, rng) {
  if (!training || p <= 0) {
    // identity in eval
    const out = new Tensor(a.shape, new Float32Array(a.data), a.requiresGrad);
    out._parents = [a];
    out._backward = () => {
      if (a.requiresGrad) { a.ensureGrad(); for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i]; }
    };
    return out;
  }
  const r = rng || Math.random;
  const mask = new Float32Array(a.size);
  const scale = 1 / (1 - p);
  const out = new Tensor(a.shape, null, a.requiresGrad);
  for (let i = 0; i < a.size; i++) {
    mask[i] = r() < p ? 0 : scale;
    out.data[i] = a.data[i] * mask[i];
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * mask[i];
  };
  return out;
}

// Embedding lookup. weights: [V, D]; ids: int[] length B. Returns [B, D].
function embedding(weights, ids) {
  assert2D(weights, 'embedding weights');
  const [V, D] = weights.shape;
  const B = ids.length;
  const out = new Tensor([B, D], null, weights.requiresGrad);
  for (let i = 0; i < B; i++) {
    const id = ids[i];
    if (id < 0 || id >= V) throw new Error(`embedding id ${id} out of range [0,${V})`);
    for (let j = 0; j < D; j++) out.data[i * D + j] = weights.data[id * D + j];
  }
  out._parents = [weights];
  out._backward = () => {
    if (!weights.requiresGrad) return;
    weights.ensureGrad();
    for (let i = 0; i < B; i++) {
      const id = ids[i];
      for (let j = 0; j < D; j++) weights.grad[id * D + j] += out.grad[i * D + j];
    }
  };
  return out;
}

// Sum all elements to a scalar.
function sumAll(a) {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  const out = new Tensor([1], new Float32Array([s]), a.requiresGrad);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[0];
  };
  return out;
}

function mean(a) {
  const s = sumAll(a);
  return mul(s, 1 / a.size);
}

module.exports = {
  Tensor,
  tensor, zeros, ones, randn, rngFromSeed,
  add, sub, mul, matmul,
  relu, leakyRelu, tanh, sigmoid, gelu, softmax,
  softmaxCrossEntropy, mseLoss,
  dropout, embedding, sumAll, mean,
  shapeSize, sameShape
};
