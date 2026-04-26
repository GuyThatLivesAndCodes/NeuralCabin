'use strict';

const { EventEmitter } = require('events');
const { trainNetwork, infer, inferStream, buildInferenceCache } = require('../engine/trainer');
const { runScript } = require('../dsl/interpreter');

// ── Structured log helper ─────────────────────────────────────────────────────

function makeLog(emit, id) {
  return (line, level = 'info') => {
    emit('log', { id, line, level, ts: Date.now() });
  };
}

// ── Backend telemetry ─────────────────────────────────────────────────────────

let _backendMeta = null;
function getBackendMeta() {
  if (_backendMeta) return _backendMeta;
  try {
    const tensor = require('../engine/tensor');
    _backendMeta = tensor.__backend || { mode: 'js', reason: 'not-checked' };
  } catch (_) {
    _backendMeta = { mode: 'unknown' };
  }
  return _backendMeta;
}

// ── TrainingManager ───────────────────────────────────────────────────────────

class TrainingManager extends EventEmitter {
  constructor(storage) {
    super();
    this.storage = storage;
    this.active = new Map();     // id -> { stop, startedAt, lastProgress }
    this._modelCache = new Map(); // id -> { updatedAt, cache }
  }

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

  backendInfo() {
    const meta = getBackendMeta();
    let extraInfo = {};
    if (meta.mode === 'rust') {
      try {
        const native = require('../../native/rust-engine/neuralcabin-node');
        extraInfo = native.api.backendInfo();
      } catch (_) {}
    }
    return { ...meta, ...extraInfo };
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
      lastProgress: null,
    };
    this.active.set(id, record);

    const log = makeLog(this.emit.bind(this), id);
    const backend = getBackendMeta();
    log(`training started (backend=${backend.mode}${backend.reason ? '/' + backend.reason : ''})`);

    (async () => {
      const t0 = Date.now();
      try {
        const result = await trainNetwork(net, {
          fromScratch,
          onProgress: (p) => {
            record.lastProgress = p;
            this.emit('progress', { id, ...p });
          },
          shouldStop: () => cancelRequested,
          log: (line) => log(line),
        });

        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        log(`training finished in ${elapsed}s (${result.metrics?.length ?? 0} epochs)`);

        this.storage.saveTrainedState(id, {
          state: result.state,
          optimizerState: result.optimizerState,
          tokenizer: result.tokenizer,
          architecture: result.architecture,
          metrics: result.metrics,
        });
        this._modelCache.delete(id);
        this.emit('done', { id, stopped: !!result.stopped, metrics: result.metrics });
      } catch (e) {
        log(`training error: ${e.message}`, 'error');
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
    makeLog(this.emit.bind(this), id)('stop requested');
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
      storage: this.storage,
    };
    return runScript(code, ctx);
  }
}

module.exports = { TrainingManager };
