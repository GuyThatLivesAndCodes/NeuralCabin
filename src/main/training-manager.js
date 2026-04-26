'use strict';

const { EventEmitter } = require('events');
const { trainNetwork, infer, inferStream, buildInferenceCache } = require('../engine/trainer');
const { runScript } = require('../dsl/interpreter');

class TrainingManager extends EventEmitter {
  constructor(storage) {
    super();
    this.storage = storage;
    this.active = new Map();      // id -> { stop, startedAt, lastProgress }
    // Model object cache: avoids rebuilding Float32Array weights + tokenizer on
    // every inference call.  Entry shape: { updatedAt: number, cache: {model, tokenizer} }
    // Invalidated when updatedAt changes (after training, restore, or update).
    this._modelCache = new Map(); // id -> { updatedAt, cache }
  }

  // Returns a { model, tokenizer } cache entry for the network, rebuilding only
  // when the network's updatedAt timestamp has changed since the last build.
  _getOrBuildModel(net) {
    const stored = this._modelCache.get(net.id);
    if (stored && stored.updatedAt === net.updatedAt) return stored.cache;
    const cache = buildInferenceCache(net);
    this._modelCache.set(net.id, { updatedAt: net.updatedAt, cache });
    return cache;
  }

  status(id) {
    const a = this.active.get(id);
    if (!a) return { running: false };
    return { running: true, startedAt: a.startedAt, lastProgress: a.lastProgress };
  }

  // opts: { fromScratch?: boolean, overrides?: object }
  async start(id, opts) {
    if (this.active.has(id)) throw new Error('Training already running');

    let net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (net.stateLocked) throw new Error('Network state is encrypted; decrypt it first.');

    opts = opts || {};
    const fromScratch = !!opts.fromScratch;
    if (opts.overrides) {
      net = { ...net, training: { ...net.training, ...opts.overrides } };
    }

    let cancelRequested = false;
    const record = {
      stop: () => { cancelRequested = true; },
      startedAt: Date.now(),
      lastProgress: null
    };
    this.active.set(id, record);

    (async () => {
      try {
        const result = await trainNetwork(net, {
          fromScratch,
          onProgress: (p) => {
            record.lastProgress = p;
            this.emit('progress', { id, ...p });
          },
          shouldStop: () => cancelRequested,
          log: (line) => this.emit('progress', { id, log: line })
        });
        this.storage.saveTrainedState(id, {
          state: result.state,
          optimizerState: result.optimizerState,
          tokenizer: result.tokenizer,
          // The trainer may have mutated arch (e.g. set vocabSize after
          // building tokenizer from corpus, or flipped isChat). Persist that
          // so the next continue-training run sees a consistent arch ↔ state
          // ↔ tokenizer triple.
          architecture: result.architecture,
          metrics: result.metrics
        });
        // Evict the stale model cache so the next inference call picks up the
        // freshly-trained weights instead of the pre-training snapshot.
        this._modelCache.delete(id);
        this.emit('done', { id, stopped: !!result.stopped, metrics: result.metrics });
      } catch (e) {
        this.emit('error', { id, message: e.message, stack: e.stack });
      } finally {
        this.active.delete(id);
      }
    })();

    return { started: true, fromScratch };
  }

  stop(id) {
    const a = this.active.get(id);
    if (!a) return { running: false };
    a.stop();
    return { stopping: true };
  }

  stopAll() { for (const [id] of this.active) this.stop(id); }

  async infer(id, input) {
    const net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (net.stateLocked) throw new Error('Network state is encrypted; decrypt it first.');
    const cache = this._getOrBuildModel(net);
    return infer(net, input, cache);
  }

  async inferStream(id, input, onToken, cancelRef) {
    const net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (net.stateLocked) throw new Error('Network state is encrypted; decrypt it first.');
    const cache = this._getOrBuildModel(net);
    return inferStream(net, input, onToken, cancelRef, cache);
  }

  async runScript(id, code) {
    const net = id ? this.storage.getNetwork(id) : null;
    const ctx = {
      network: net,
      saveTrainedState: (state) => net && this.storage.saveTrainedState(net.id, state),
      storage: this.storage
    };
    return runScript(code, ctx);
  }
}

module.exports = { TrainingManager };
