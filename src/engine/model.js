'use strict';

const T = require('./tensor');
const { Sequential, Linear, Activation, Dropout, Embedding, buildLayer, restoreLayer } = require('./layers');

// Build a Sequential from a high-level spec describing a flat model.
// architecture:
//   kind: 'mlp' | 'classifier' | 'regressor' | 'charLM'
//   inputDim, outputDim, hidden:[int], activation:'relu'|..., dropout:0..1
//   for charLM: vocabSize, embDim, contextLen, hidden:[int]
function buildModel(arch, rng) {
  const layers = [];
  const act = arch.activation || 'relu';
  const drop = arch.dropout || 0;

  if (arch.kind === 'mlp' || arch.kind === 'classifier' || arch.kind === 'regressor') {
    const sizes = [arch.inputDim, ...(arch.hidden || []), arch.outputDim];
    for (let i = 0; i < sizes.length - 1; i++) {
      layers.push(new Linear(sizes[i], sizes[i + 1], rng));
      const isLast = i === sizes.length - 2;
      if (!isLast) {
        layers.push(new Activation(act));
        if (drop > 0) layers.push(new Dropout(drop));
      } else if (arch.kind === 'classifier' && arch.softmaxOutput) {
        // Usually we apply softmax+CE externally; skip here.
      }
    }
    return new Sequential(layers);
  }

  if (arch.kind === 'charLM') {
    // Embedding → flatten (contextLen*embDim) → MLP → vocabSize logits
    const ctx = arch.contextLen;
    const emb = arch.embDim;
    layers.push(new Embedding(arch.vocabSize, emb, rng));
    // After embedding we get [B, ctx*emb] via concat — but our embedding returns [B,emb] for ids[B].
    // For charLM we treat the "batch" dim as B*ctx then reshape OUTSIDE the sequential via a wrapper.
    // Instead we build: reshape done in inference code, sequential just handles flat [B, ctx*emb] → logits.
    // So the embedding layer here is used directly by the CharLM wrapper; leave sequential as the rest.
    const flat = ctx * emb;
    const sizes = [flat, ...(arch.hidden || []), arch.vocabSize];
    for (let i = 0; i < sizes.length - 1; i++) {
      layers.push(new Linear(sizes[i], sizes[i + 1], rng));
      const isLast = i === sizes.length - 2;
      if (!isLast) {
        layers.push(new Activation(act));
        if (drop > 0) layers.push(new Dropout(drop));
      }
    }
    return new Sequential(layers);
  }

  throw new Error('unknown arch kind: ' + arch.kind);
}

// Char-level LM helper that owns an Embedding layer separately and reshapes.
class CharLM {
  constructor(arch, rng) {
    this.arch = arch;
    this.embedding = new Embedding(arch.vocabSize, arch.embDim, rng);
    const flat = arch.contextLen * arch.embDim;
    const layers = [];
    const sizes = [flat, ...(arch.hidden || []), arch.vocabSize];
    const act = arch.activation || 'relu';
    for (let i = 0; i < sizes.length - 1; i++) {
      layers.push(new Linear(sizes[i], sizes[i + 1], rng));
      if (i < sizes.length - 2) {
        layers.push(new Activation(act));
        if (arch.dropout) layers.push(new Dropout(arch.dropout));
      }
    }
    this.head = new Sequential(layers);
  }
  get params() { return [...this.embedding.params, ...this.head.params]; }

  // idsBatch: int[][] shape [B][contextLen]
  forward(idsBatch, ctx) {
    const B = idsBatch.length;
    const L = this.arch.contextLen;
    const D = this.arch.embDim;
    // Build flat id array of length B*L, then embed to [B*L, D], then reshape to [B, L*D]
    const flatIds = new Array(B * L);
    for (let i = 0; i < B; i++) for (let j = 0; j < L; j++) flatIds[i * L + j] = idsBatch[i][j];
    const embOut = this.embedding.forward(flatIds); // [B*L, D]
    // Create a reshape "view" — since data is already laid out row-major [B*L, D],
    // it IS [B, L*D] with the same memory. We can build a Tensor pointing to same data
    // but with a custom backward that maps grads back.
    const reshaped = new T.Tensor([B, L * D], embOut.data, embOut.requiresGrad);
    reshaped._parents = [embOut];
    reshaped._backward = () => {
      if (!embOut.requiresGrad) return;
      embOut.ensureGrad();
      for (let i = 0; i < embOut.size; i++) embOut.grad[i] += reshaped.grad[i];
    };
    // Share the same grad buffer by aliasing on .ensureGrad — simpler: force them to share
    reshaped.ensureGrad = function () {
      if (!reshaped.grad) reshaped.grad = new Float32Array(reshaped.size);
      return reshaped.grad;
    };
    return this.head.forward(reshaped, ctx);
  }

  toJSON() {
    return {
      kind: 'charLM',
      arch: this.arch,
      embedding: this.embedding.toJSON(),
      head: this.head.toJSON()
    };
  }

  static fromJSON(obj, rng) {
    const m = new CharLM(obj.arch, rng);
    m.embedding.loadFromJSON(obj.embedding);
    m.head = Sequential.fromJSON(obj.head, rng);
    return m;
  }
}

function buildFromSpec(spec, rng) {
  if (spec.kind === 'charLM') return new CharLM(spec, rng);
  return buildModel(spec, rng);
}

function restoreFromState(state, spec, rng) {
  // state.kind tells us
  if (state.kind === 'charLM') return CharLM.fromJSON(state, rng);
  return Sequential.fromJSON(state, rng);
}

module.exports = { buildFromSpec, restoreFromState, CharLM };
