'use strict';
// Neuroevolution — Selective Reproduction.
// Rust handles crossover/mutation/selection arithmetic; this module manages
// the population of JS model objects and orchestrates the evolutionary loop.

const T = require('./tensor');
const { buildFromSpec, restoreFromState } = require('./model');

function getRust() {
  const backend = T.__backend;
  if (backend && backend.mode === 'rust') {
    try { return require('../../native/rust-engine/neuralcabin-node').api.rust; } catch (_) {}
  }
  return null;
}

// ── Population ────────────────────────────────────────────────────────────────

class Population {
  /**
   * opts = {
   *   architecture,          // NeuralCabin arch spec (same as for buildFromSpec)
   *   size: 50,              // number of individuals
   *   eliteCount: 5,         // top survivors copied unchanged each generation
   *   pMutate: 0.1,          // per-weight mutation probability
   *   mutationStd: 0.02,     // Gaussian std for mutations
   *   tournamentK: 3,        // tournament size for parent selection
   *   seed: 42,
   * }
   */
  constructor(opts = {}) {
    this.arch = opts.architecture ?? { kind: 'classifier', inputDim: 4, outputDim: 2, hidden: [32] };
    this.size = opts.size ?? 50;
    this.eliteCount = opts.eliteCount ?? 5;
    this.pMutate = opts.pMutate ?? 0.1;
    this.mutationStd = opts.mutationStd ?? 0.02;
    this.tournamentK = opts.tournamentK ?? 3;
    this.seed = opts.seed ?? 42;
    this.generation = 0;
    this.bestFitness = -Infinity;
    this.bestIndividual = null;
    this.history = []; // { generation, stats } per generation

    // Build initial population with independent random seeds.
    this.individuals = [];
    for (let i = 0; i < this.size; i++) {
      const rng = T.rngFromSeed(this.seed + i * 7919);
      this.individuals.push(buildFromSpec(this.arch, rng));
    }
    this.fitnesses = new Float32Array(this.size);
  }

  // ── Evaluation ─────────────────────────────────────────────────────────────

  /**
   * Evaluate fitness for all individuals.
   * fitnessFn(model, index) → number  (higher is better)
   * Returns this for chaining.
   */
  evaluate(fitnessFn) {
    for (let i = 0; i < this.size; i++) {
      this.fitnesses[i] = fitnessFn(this.individuals[i], i);
    }
    // Track best
    for (let i = 0; i < this.size; i++) {
      if (this.fitnesses[i] > this.bestFitness) {
        this.bestFitness = this.fitnesses[i];
        this.bestIndividual = this.individuals[i];
      }
    }
    return this;
  }

  // ── Evolution step ─────────────────────────────────────────────────────────

  /**
   * Run one generation of evolution.
   * Requires evaluate() to have been called first.
   * Returns fitness stats { min, max, mean, std }.
   */
  evolve() {
    const rust = getRust();
    const paramCount = this._paramCount();
    const fitArr = Array.from(this.fitnesses);
    const genSeed = (this.seed + this.generation * 100003) >>> 0;

    let nextParams;
    if (rust && paramCount > 0) {
      // Pack all individual param vectors into one flat array.
      const flat = new Float32Array(this.size * paramCount);
      for (let i = 0; i < this.size; i++) {
        flat.set(this._getParams(this.individuals[i]), i * paramCount);
      }
      nextParams = rust.neEvolveGeneration(
        Array.from(flat), fitArr, paramCount,
        this.eliteCount, this.pMutate, this.mutationStd, this.tournamentK, genSeed
      );
    } else {
      nextParams = this._evolveJs(fitArr, paramCount, genSeed);
    }

    // Unpack new params back into individual model objects.
    const nextIndividuals = [];
    for (let i = 0; i < this.size; i++) {
      const p = nextParams.slice(i * paramCount, (i + 1) * paramCount);
      const rng = T.rngFromSeed(genSeed + i * 1000003);
      const model = buildFromSpec(this.arch, rng);
      this._setParams(model, new Float32Array(p));
      nextIndividuals.push(model);
    }

    // Record stats before wiping fitnesses.
    const stats = rust
      ? rust.neFitnessStats(fitArr)
      : this._statsJs(fitArr);
    this.history.push({ generation: this.generation, stats });

    this.individuals = nextIndividuals;
    this.fitnesses.fill(0);
    this.generation++;
    return stats;
  }

  // ── Convenience helpers ────────────────────────────────────────────────────

  getBest() {
    let best = 0;
    for (let i = 1; i < this.size; i++) if (this.fitnesses[i] > this.fitnesses[best]) best = i;
    return this.individuals[best];
  }

  // ── Serialization ──────────────────────────────────────────────────────────

  toJSON() {
    return {
      arch: this.arch,
      size: this.size,
      eliteCount: this.eliteCount,
      pMutate: this.pMutate,
      mutationStd: this.mutationStd,
      tournamentK: this.tournamentK,
      seed: this.seed,
      generation: this.generation,
      bestFitness: this.bestFitness,
      history: this.history,
      individuals: this.individuals.map(m => m.toJSON()),
      fitnesses: Array.from(this.fitnesses),
    };
  }

  static fromJSON(obj) {
    const pop = new Population({ architecture: obj.arch, size: obj.size, seed: obj.seed });
    pop.eliteCount = obj.eliteCount;
    pop.pMutate = obj.pMutate;
    pop.mutationStd = obj.mutationStd;
    pop.tournamentK = obj.tournamentK;
    pop.generation = obj.generation ?? 0;
    pop.bestFitness = obj.bestFitness ?? -Infinity;
    pop.history = obj.history ?? [];
    const rng = T.rngFromSeed(obj.seed);
    pop.individuals = obj.individuals.map(s => restoreFromState(s, obj.arch, rng));
    pop.fitnesses = new Float32Array(obj.fitnesses ?? pop.size);
    return pop;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  _paramCount() {
    if (this.individuals.length === 0) return 0;
    return this.individuals[0].params.reduce((s, p) => s + p.size, 0);
  }

  _getParams(model) {
    const total = model.params.reduce((s, p) => s + p.size, 0);
    const flat = new Float32Array(total);
    let offset = 0;
    for (const p of model.params) { flat.set(p.data, offset); offset += p.size; }
    return flat;
  }

  _setParams(model, flat) {
    let offset = 0;
    for (const p of model.params) { p.data.set(flat.subarray(offset, offset + p.size)); offset += p.size; }
  }

  _evolveJs(fitnesses, paramCount, seed) {
    const pop = this.size;
    // Sort descending for elites
    const sorted = fitnesses.map((f, i) => ({ f, i })).sort((a, b) => b.f - a.f);
    const next = new Array(pop * paramCount);
    let rngS = seed;

    const getRng = () => {
      rngS = (rngS + 0x6D2B79F5) >>> 0;
      let t = rngS;
      t = (t ^ (t >>> 15)) * (t | 1) >>> 0;
      t ^= t + ((t ^ (t >>> 7)) * (t | 61) >>> 0) >>> 0;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    for (let i = 0; i < pop; i++) {
      const dst = next.slice ? next : next; // keep reference
      const p1 = sorted[i < this.eliteCount ? i : Math.floor(getRng() * Math.min(pop, 10))].i;
      const p2 = sorted[Math.floor(getRng() * Math.min(pop, 10))].i;
      const s1 = this._getParams(this.individuals[p1]);
      const s2 = this._getParams(this.individuals[p2]);
      for (let j = 0; j < paramCount; j++) {
        let v = getRng() < 0.5 ? s1[j] : s2[j];
        if (i >= this.eliteCount && getRng() < this.pMutate) {
          // Box-Muller normal sample
          const u = Math.max(1e-10, getRng()), vv = Math.max(1e-10, getRng());
          v += Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * vv) * this.mutationStd;
        }
        next[i * paramCount + j] = v;
      }
    }
    return next;
  }

  _statsJs(fitnesses) {
    const n = fitnesses.length;
    const min = Math.min(...fitnesses);
    const max = Math.max(...fitnesses);
    const mean = fitnesses.reduce((s, f) => s + f, 0) / n;
    const std = Math.sqrt(fitnesses.reduce((s, f) => s + (f - mean) ** 2, 0) / n);
    return { min, max, mean, std };
  }
}

// ── Convenience: run an evolution loop ───────────────────────────────────────

/**
 * Run a complete evolutionary training session.
 * fitnessFn(model, index) → number
 * hooks = { onGeneration({ generation, stats, bestFitness }), shouldStop() }
 */
async function evolveNetwork(population, generations, fitnessFn, hooks = {}) {
  const onGen = hooks.onGeneration || (() => {});
  const shouldStop = hooks.shouldStop || (() => false);
  const yieldFn = () => new Promise(r => setImmediate(r));

  for (let g = 0; g < generations; g++) {
    if (shouldStop()) break;
    population.evaluate(fitnessFn);
    const stats = population.evolve();
    onGen({ generation: population.generation, stats, bestFitness: population.bestFitness });
    await yieldFn();
  }

  return {
    generation: population.generation,
    bestFitness: population.bestFitness,
    best: population.getBest(),
    history: population.history,
  };
}

module.exports = { Population, evolveNetwork };
