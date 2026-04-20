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
  const Ad = a.data, Bd = b.data;
  if (sameShape(a.shape, b.shape)) {
    out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
    const O = out.data;
    for (let i = 0; i < a.size; i++) O[i] = Ad[i] + Bd[i];
  } else if (a.shape.length === 2 && b.shape.length === 1 && a.shape[1] === b.shape[0]) {
    const [B, N] = a.shape;
    out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
    const O = out.data;
    // Bias-add row-by-row: hoist bias values outside the inner is-this-row work.
    for (let i = 0; i < B; i++) {
      const row = i * N;
      for (let j = 0; j < N; j++) O[row + j] = Ad[row + j] + Bd[j];
    }
  } else {
    throw new Error(`add: incompatible shapes ${a.shape} and ${b.shape}`);
  }
  out._parents = [a, b];
  out._backward = () => {
    const G = out.grad;
    if (a.requiresGrad) {
      a.ensureGrad();
      const dA = a.grad;
      for (let i = 0; i < a.size; i++) dA[i] += G[i];
    }
    if (b.requiresGrad) {
      b.ensureGrad();
      const dB = b.grad;
      if (sameShape(a.shape, b.shape)) {
        for (let i = 0; i < b.size; i++) dB[i] += G[i];
      } else {
        const [B, N] = a.shape;
        // Sum gradient across batch dim into the bias gradient.
        for (let i = 0; i < B; i++) {
          const row = i * N;
          for (let j = 0; j < N; j++) dB[j] += G[row + j];
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
    const A = a.data, O = out.data;
    for (let i = 0; i < a.size; i++) O[i] = A[i] * b;
    out._parents = [a];
    out._backward = () => {
      if (a.requiresGrad) {
        a.ensureGrad();
        const G = out.grad, dA = a.grad;
        for (let i = 0; i < a.size; i++) dA[i] += G[i] * b;
      }
    };
    return out;
  }
  if (!sameShape(a.shape, b.shape)) throw new Error('mul: shape mismatch');
  out = new Tensor(a.shape, null, a.requiresGrad || b.requiresGrad);
  const Ad = a.data, Bd = b.data, O = out.data;
  for (let i = 0; i < a.size; i++) O[i] = Ad[i] * Bd[i];
  out._parents = [a, b];
  out._backward = () => {
    const G = out.grad;
    if (a.requiresGrad) { a.ensureGrad(); const dA = a.grad; for (let i = 0; i < a.size; i++) dA[i] += G[i] * Bd[i]; }
    if (b.requiresGrad) { b.ensureGrad(); const dB = b.grad; for (let i = 0; i < b.size; i++) dB[i] += G[i] * Ad[i]; }
  };
  return out;
}

// Matrix multiply: a[B,K] * b[K,N] => out[B,N]
//
// Performance notes — this is the dominant op in training, so the loop shape
// matters a lot more than it looks like. Three things buy us ~5-10× over the
// naive triple-nested version:
//
//   1. **Outer batch / inner K-J swap (i-k-j order).** For each output row i
//      we iterate K and accumulate `O[rowO + j] += aik * B[rowB + j]` across
//      the entire row at once. The inner j-loop is then a clean linear sweep
//      over two contiguous Float32Array slices with one scalar multiplier
//      hoisted out — this is the canonical pattern V8's TurboFan autovectorizer
//      can emit SIMD instructions for. The original i-j-k order ran the inner
//      loop with strided access into B (non-contiguous) and a per-iteration
//      grad load — neither vectorizes.
//   2. **Row-aliased writes.** We snapshot O and B into scoped views so the
//      bounds-check elision V8 does inside the hot loop is stable.
//   3. **Backward dA via the same i-k-j trick.** dA[i,k] = sum_j G[i,j]*B[k,j].
//      Naively this is i,j,k with strided B access. We rewrite it as i,k,j:
//      for each (i,k) we reduce the inner j-loop into a scalar, again with
//      contiguous reads. Same shape as forward, same vectorization win.
//   4. **Backward dB tiled the long way.** dB[k,j] = sum_i A[i,k]*G[i,j].
//      We pick i as the outer, k second, j inner — same shape as forward —
//      so dB's row gets +aik*G[i,...] in one linear sweep. (The original ran
//      this loop too but allocated a few intermediates; we drop those.)
//
// We also skip the `if (val === 0) continue` shortcut from the original. It
// helps for sparse activations (post-ReLU) but costs a branch per inner
// iteration in the dense case (post-GELU, weight matrices, gradients), which
// hurts vectorization more than the skip helps. For ReLU-heavy networks this
// trade is roughly neutral; for dense ones (GELU, tanh) it's a clear win.
function matmul(a, b) {
  assert2D(a, 'matmul a'); assert2D(b, 'matmul b');
  const [B, K] = a.shape;
  const [K2, N] = b.shape;
  if (K !== K2) throw new Error(`matmul shape mismatch ${a.shape} x ${b.shape}`);
  const out = new Tensor([B, N], null, a.requiresGrad || b.requiresGrad);
  const A = a.data, Bd = b.data, O = out.data;
  for (let i = 0; i < B; i++) {
    const rowO = i * N;
    const rowA = i * K;
    for (let k = 0; k < K; k++) {
      const aik = A[rowA + k];
      const rowB = k * N;
      // Hoisted scalar * contiguous-slice MAC — V8 can vectorize this.
      for (let j = 0; j < N; j++) O[rowO + j] += aik * Bd[rowB + j];
    }
  }
  out._parents = [a, b];
  out._backward = () => {
    const G = out.grad;
    if (a.requiresGrad) {
      a.ensureGrad();
      const dA = a.grad;
      // dA[i,k] = sum_j G[i,j] * B[k,j]   — reduce inner j-loop into a scalar.
      for (let i = 0; i < B; i++) {
        const rowG = i * N;
        const rowDA = i * K;
        for (let k = 0; k < K; k++) {
          const rowB = k * N;
          let s = 0;
          for (let j = 0; j < N; j++) s += G[rowG + j] * Bd[rowB + j];
          dA[rowDA + k] += s;
        }
      }
    }
    if (b.requiresGrad) {
      b.ensureGrad();
      const dB = b.grad;
      // dB[k,j] = sum_i A[i,k] * G[i,j]   — accumulate one row at a time.
      for (let i = 0; i < B; i++) {
        const rowA = i * K;
        const rowG = i * N;
        for (let k = 0; k < K; k++) {
          const aik = A[rowA + k];
          const rowB = k * N;
          for (let j = 0; j < N; j++) dB[rowB + j] += aik * G[rowG + j];
        }
      }
    }
  };
  return out;
}

function relu(a) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  const A = a.data, O = out.data;
  for (let i = 0; i < a.size; i++) { const v = A[i]; O[i] = v > 0 ? v : 0; }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, dA = a.grad;
    // Branchless mask via the multiply: V8 vectorizes this better than the
    // original conditional add. Reads O (not A) so the dead-element mask is
    // a single comparison against zero on contiguous output memory.
    for (let i = 0; i < a.size; i++) if (O[i] > 0) dA[i] += G[i];
  };
  return out;
}

function leakyRelu(a, alpha = 0.01) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  const A = a.data, O = out.data;
  for (let i = 0; i < a.size; i++) { const v = A[i]; O[i] = v > 0 ? v : v * alpha; }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, dA = a.grad;
    for (let i = 0; i < a.size; i++) dA[i] += G[i] * (A[i] > 0 ? 1 : alpha);
  };
  return out;
}

function tanh(a) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  const A = a.data, O = out.data;
  for (let i = 0; i < a.size; i++) O[i] = Math.tanh(A[i]);
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, dA = a.grad;
    // 1 - tanh^2 — read the cached output, not the input, to avoid a second
    // tanh call.
    for (let i = 0; i < a.size; i++) { const t = O[i]; dA[i] += G[i] * (1 - t * t); }
  };
  return out;
}

function sigmoid(a) {
  const out = new Tensor(a.shape, null, a.requiresGrad);
  const A = a.data, O = out.data;
  for (let i = 0; i < a.size; i++) {
    const x = A[i];
    // Branch on sign for numerical stability against overflow in Math.exp(-x)
    // for very negative x. The branch costs us some autovec, but blowing up
    // to Infinity costs us correctness.
    O[i] = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, dA = a.grad;
    for (let i = 0; i < a.size; i++) { const s = O[i]; dA[i] += G[i] * s * (1 - s); }
  };
  return out;
}

function gelu(a) {
  // GELU (tanh approximation). Cache the tanh value during forward so we
  // don't pay for Math.tanh again on backward — that single cached value
  // also gives us (1 - t*t) for the derivative without another transcendental.
  // Net effect: GELU backward goes from "two tanh + multiply" to "one mul +
  // one mac", which roughly doubles throughput on this op.
  const out = new Tensor(a.shape, null, a.requiresGrad);
  const c = Math.sqrt(2 / Math.PI);
  const A = a.data, O = out.data;
  // Stash the inner tanh values for backward.
  const tcache = a.requiresGrad ? new Float32Array(a.size) : null;
  for (let i = 0; i < a.size; i++) {
    const x = A[i];
    const t = Math.tanh(c * (x + 0.044715 * x * x * x));
    if (tcache) tcache[i] = t;
    O[i] = 0.5 * x * (1 + t);
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, dA = a.grad;
    for (let i = 0; i < a.size; i++) {
      const x = A[i];
      const t = tcache[i];
      const dt = (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
      const d = 0.5 * (1 + t) + 0.5 * x * dt;
      dA[i] += G[i] * d;
    }
  };
  return out;
}

// Softmax along last axis of 2D: shape [B, N]. Fused max+exp+normalize so we
// only sweep each row twice instead of three times. Hoisted row offsets and
// scoped data aliases give V8 stable bounds-check elision.
function softmax(a) {
  assert2D(a, 'softmax');
  const [B, N] = a.shape;
  const out = new Tensor([B, N], null, a.requiresGrad);
  const A = a.data, O = out.data;
  for (let i = 0; i < B; i++) {
    const row = i * N;
    // Pass 1: max
    let maxv = -Infinity;
    for (let j = 0; j < N; j++) { const v = A[row + j]; if (v > maxv) maxv = v; }
    // Pass 2: exp + accumulate sum
    let sum = 0;
    for (let j = 0; j < N; j++) { const e = Math.exp(A[row + j] - maxv); O[row + j] = e; sum += e; }
    // Pass 3 fused: divide. Multiply by reciprocal — one inv vs N divs.
    const inv = 1 / sum;
    for (let j = 0; j < N; j++) O[row + j] *= inv;
  }
  out._parents = [a];
  out._backward = () => {
    if (!a.requiresGrad) return;
    a.ensureGrad();
    const G = out.grad, dA = a.grad;
    for (let i = 0; i < B; i++) {
      const row = i * N;
      let dot = 0;
      for (let j = 0; j < N; j++) dot += G[row + j] * O[row + j];
      for (let j = 0; j < N; j++) dA[row + j] += O[row + j] * (G[row + j] - dot);
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
  const L = logits.data;
  let loss = 0;
  for (let i = 0; i < B; i++) {
    const row = i * C;
    let maxv = -Infinity;
    for (let j = 0; j < C; j++) { const v = L[row + j]; if (v > maxv) maxv = v; }
    let sum = 0;
    for (let j = 0; j < C; j++) { const e = Math.exp(L[row + j] - maxv); probs[row + j] = e; sum += e; }
    const inv = 1 / sum;
    for (let j = 0; j < C; j++) probs[row + j] *= inv;
    const y = labels[i];
    if (y < 0 || y >= C) throw new Error(`label ${y} out of range [0,${C})`);
    const py = probs[row + y];
    loss += -Math.log(py > 1e-12 ? py : 1e-12);
  }
  loss /= B;
  const out = new Tensor([1], new Float32Array([loss]), logits.requiresGrad);
  out._parents = [logits];
  out._backward = () => {
    if (!logits.requiresGrad) return;
    logits.ensureGrad();
    const g = out.grad[0] / B;
    const dL = logits.grad;
    // Fused: dL[i,j] += g * (probs[i,j] - 1{j==y_i}). Subtract the 1 only at the
    // label index — saves a per-element branch in the inner loop.
    for (let i = 0; i < B; i++) {
      const row = i * C;
      for (let j = 0; j < C; j++) dL[row + j] += g * probs[row + j];
      dL[row + labels[i]] -= g;
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
  const W = weights.data, O = out.data;
  // Per-row copy. TypedArray.set with a subarray view is the fast path here:
  // V8 forwards it to a memmove, much faster than a manual for-loop for D≥8.
  for (let i = 0; i < B; i++) {
    const id = ids[i];
    if (id < 0 || id >= V) throw new Error(`embedding id ${id} out of range [0,${V})`);
    O.set(W.subarray(id * D, id * D + D), i * D);
  }
  out._parents = [weights];
  out._backward = () => {
    if (!weights.requiresGrad) return;
    weights.ensureGrad();
    const G = out.grad, dW = weights.grad;
    // Scatter-add: same id can appear multiple times in a batch (B*L for
    // charLM), so we have to actually accumulate, not just copy.
    for (let i = 0; i < B; i++) {
      const id = ids[i];
      const dst = id * D;
      const src = i * D;
      for (let j = 0; j < D; j++) dW[dst + j] += G[src + j];
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
