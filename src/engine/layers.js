'use strict';

const T = require('./tensor');

// Every layer has: .params (Tensor[] with requiresGrad=true), .forward(x), .toJSON(), .loadFromJSON()

class Linear {
  // in: input features, out: output features
  constructor(inFeat, outFeat, rng) {
    this.type = 'Linear';
    this.inFeat = inFeat;
    this.outFeat = outFeat;
    // Kaiming uniform-ish init
    const std = Math.sqrt(2 / inFeat);
    const wData = new Float32Array(inFeat * outFeat);
    const r = rng || Math.random;
    for (let i = 0; i < wData.length; i += 2) {
      let u = 0, v = 0;
      while (u === 0) u = r();
      while (v === 0) v = r();
      const mag = Math.sqrt(-2.0 * Math.log(u)) * std;
      wData[i] = mag * Math.cos(2 * Math.PI * v);
      if (i + 1 < wData.length) wData[i + 1] = mag * Math.sin(2 * Math.PI * v);
    }
    this.W = new T.Tensor([inFeat, outFeat], wData, true);
    this.b = new T.Tensor([outFeat], new Float32Array(outFeat), true);
    this.params = [this.W, this.b];
  }
  forward(x) { return T.add(T.matmul(x, this.W), this.b); }
  toJSON() { return { type: this.type, inFeat: this.inFeat, outFeat: this.outFeat, W: this.W.toJSON(), b: this.b.toJSON() }; }
  loadFromJSON(o) {
    this.W.data = new Float32Array(o.W.data);
    this.b.data = new Float32Array(o.b.data);
  }
}

class Activation {
  constructor(kind) {
    this.type = 'Activation';
    this.kind = kind; // relu | leakyRelu | tanh | sigmoid | gelu | softmax
    this.params = [];
  }
  forward(x) {
    switch (this.kind) {
      case 'relu': return T.relu(x);
      case 'leakyRelu': return T.leakyRelu(x);
      case 'tanh': return T.tanh(x);
      case 'sigmoid': return T.sigmoid(x);
      case 'gelu': return T.gelu(x);
      case 'softmax': return T.softmax(x);
      default: throw new Error('unknown activation ' + this.kind);
    }
  }
  toJSON() { return { type: this.type, kind: this.kind }; }
  loadFromJSON() {}
}

class Dropout {
  constructor(p) {
    this.type = 'Dropout';
    this.p = p;
    this.params = [];
  }
  forward(x, ctx) { return T.dropout(x, this.p, ctx.training, ctx.rng); }
  toJSON() { return { type: this.type, p: this.p }; }
  loadFromJSON() {}
}

class Embedding {
  constructor(vocabSize, dim, rng) {
    this.type = 'Embedding';
    this.vocabSize = vocabSize;
    this.dim = dim;
    const data = new Float32Array(vocabSize * dim);
    const r = rng || Math.random;
    for (let i = 0; i < data.length; i++) data[i] = (r() - 0.5) * 0.1;
    this.W = new T.Tensor([vocabSize, dim], data, true);
    this.params = [this.W];
  }
  forward(idsTensorOrArray) {
    const ids = Array.isArray(idsTensorOrArray) ? idsTensorOrArray : Array.from(idsTensorOrArray.data).map(v => v | 0);
    return T.embedding(this.W, ids);
  }
  toJSON() { return { type: this.type, vocabSize: this.vocabSize, dim: this.dim, W: this.W.toJSON() }; }
  loadFromJSON(o) { this.W.data = new Float32Array(o.W.data); }
}

function buildLayer(spec, rng) {
  switch (spec.type) {
    case 'Linear': return new Linear(spec.inFeat, spec.outFeat, rng);
    case 'Activation': return new Activation(spec.kind);
    case 'Dropout': return new Dropout(spec.p);
    case 'Embedding': return new Embedding(spec.vocabSize, spec.dim, rng);
    default: throw new Error('unknown layer ' + spec.type);
  }
}

function restoreLayer(obj, rng) {
  const l = buildLayer(obj, rng);
  l.loadFromJSON(obj);
  return l;
}

// Simple sequential container.
class Sequential {
  constructor(layers) {
    this.layers = layers || [];
  }
  get params() {
    const p = [];
    for (const l of this.layers) for (const t of (l.params || [])) p.push(t);
    return p;
  }
  forward(x, ctx) {
    let h = x;
    ctx = ctx || { training: false };
    for (const l of this.layers) {
      h = l.forward(h, ctx);
    }
    return h;
  }
  toJSON() { return { layers: this.layers.map(l => l.toJSON()) }; }
  static fromJSON(obj, rng) {
    return new Sequential(obj.layers.map(o => restoreLayer(o, rng)));
  }
}

module.exports = { Linear, Activation, Dropout, Embedding, Sequential, buildLayer, restoreLayer };
