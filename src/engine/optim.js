'use strict';

class SGD {
  constructor(params, { lr = 0.01, momentum = 0 } = {}) {
    this.params = params;
    this.lr = lr;
    this.momentum = momentum;
    this.v = params.map(p => new Float32Array(p.size));
  }
  step() {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const v = this.v[i];
      for (let j = 0; j < p.size; j++) {
        v[j] = this.momentum * v[j] + p.grad[j];
        p.data[j] -= this.lr * v[j];
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }

  // Serialize the velocity buffers so training can resume without losing momentum.
  // Buffers are matched to params positionally — if the model architecture changes
  // we'll detect a length mismatch on load and refuse, falling back to a fresh state.
  toJSON() {
    return {
      type: 'sgd',
      lr: this.lr,
      momentum: this.momentum,
      v: this.v.map(buf => Array.from(buf))
    };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'sgd') return false;
    if (!Array.isArray(o.v) || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) this.v[i] = new Float32Array(o.v[i]);
    return true;
  }
}

class Adam {
  constructor(params, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
    this.t = 0;
  }
  step() {
    this.t++;
    const { beta1, beta2, eps, lr } = this;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i], v = this.v[i];
      for (let j = 0; j < p.size; j++) {
        let g = p.grad[j];
        if (this.weightDecay) g += this.weightDecay * p.data[j];
        m[j] = beta1 * m[j] + (1 - beta1) * g;
        v[j] = beta2 * v[j] + (1 - beta2) * g * g;
        const mhat = m[j] / bc1;
        const vhat = v[j] / bc2;
        p.data[j] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }

  // Serialize so we can resume training mid-trajectory without an Adam "warmup
  // bump" — the momentum (m) and variance (v) buffers carry the optimizer's
  // running statistics. The step counter t is needed for bias correction.
  toJSON() {
    return {
      type: 'adam',
      lr: this.lr, beta1: this.beta1, beta2: this.beta2, eps: this.eps,
      weightDecay: this.weightDecay, t: this.t,
      m: this.m.map(buf => Array.from(buf)),
      v: this.v.map(buf => Array.from(buf))
    };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'adam') return false;
    if (!Array.isArray(o.m) || !Array.isArray(o.v)) return false;
    if (o.m.length !== this.params.length || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) {
      this.m[i] = new Float32Array(o.m[i]);
      this.v[i] = new Float32Array(o.v[i]);
    }
    this.t = o.t || 0;
    return true;
  }
}

function buildOptim(name, params, opts) {
  if (name === 'sgd') return new SGD(params, opts);
  if (name === 'adam') return new Adam(params, opts);
  throw new Error('unknown optimizer ' + name);
}

module.exports = { SGD, Adam, buildOptim };
