'use strict';

const path = require('path');
const { app } = require('electron');

// Load neuroevolution + tensor from the app bundle.
let Population, T;
try {
  ({ Population } = require(path.join(app.getAppPath(), 'src', 'engine', 'neuroevolution')));
  T = require(path.join(app.getAppPath(), 'src', 'engine', 'tensor'));
} catch (e) {
  console.error('[self-driving-car] Failed to load engines:', e.message);
}

// ── Simulation constants ──────────────────────────────────────────────────────
const DEFAULT_POP_SIZE = 20;
const INPUT_DIM   = 11;   // 9 sensor rays + normalised speed + steer_memory
const OUTPUT_DIM  = 2;    // steer (-1 to 1), throttle/brake (-1 to 1)
const TRACK_PTS   = 8;    // Base control points before Chaikin smoothing
const HALF_W      = 28;   // Track half-width for collision detection
const MAX_FRAMES  = (30 * 18);  // ~18 s episode limit

const RAY_ANGLES  = Array.from({length: 9}, (_, i) => -1.4 + (i * 2.8 / 14));
const RAY_MAX     = 40;
const RAY_STEP    = 5;

const DT          = 1 / 30;
const MAX_SPEED   = 340;
const ACCEL       = 230;
const BRAKE_FORCE = 350;    // Stronger than acceleration for realistic braking
const FRICTION    = 0.93;   // Multiplicative drag applied every frame
const STEER_RATE  = 2.4;
const GRIP_LOSS   = 0.4;    // Speed lost during hard cornering (proportional to steer × speed)

const CANVAS_CX   = 350;
const CANVAS_CY   = 285;

// ── Module-level state ────────────────────────────────────────────────────────
let _pop        = null;
let _popSize    = DEFAULT_POP_SIZE;
let _maxGens    = 0;       // 0 = unlimited
let _track      = null;
let _cars       = [];
let _carFit     = [];
let _generation = 0;
let _bestFit    = 0;
let _genBestFit = 0;
let _running    = false;
let _genHistory = [];      // [{gen, best, mean}]

// ── Inference state ───────────────────────────────────────────────────────────
let _inferCar   = null;
let _inferTrack = null;

// ── LCG seeded random ─────────────────────────────────────────────────────────
function lcg(s) { return (Math.imul(s | 0, 1664525) + 1013904223) >>> 0; }
function lcgFloat(s) { return [(lcg(s) >>> 0) / 4294967296, lcg(s)]; }

// ── Track generation ──────────────────────────────────────────────────────────

function generateTrack(seed) {
  let rng = (seed || 0xDEADBEEF) >>> 0;

  function nextRand() {
    let f; [f, rng] = lcgFloat(rng);
    return f;
  }

  const initialN  = TRACK_PTS * 2;
  const baseRadius = 200;
  const complexity = Math.floor(nextRand() * 3) + 2;
  const phase      = nextRand() * Math.PI * 2;
  const lobeDepth  = 40 + nextRand() * 60;
  const jitter     = 0.15;

  const angles = Array.from({ length: initialN }, (_, i) => (i / initialN) * 2 * Math.PI);

  const radii = angles.map((a) => {
    let r = baseRadius + Math.sin(a * complexity + phase) * lobeDepth;
    r += (nextRand() - 0.5) * baseRadius * jitter;
    return Math.max(50, Math.min(230, r));
  });

  let pts = angles.map((a, i) => [
    CANVAS_CX + Math.cos(a) * radii[i],
    CANVAS_CY + Math.sin(a) * radii[i],
  ]);

  // True Chaikin corner-cutting — 4 passes doubles point count each time
  for (let iter = 0; iter < 4; iter++) {
    const newPts = [];
    for (let i = 0; i < pts.length; i++) {
      const p1 = pts[i], p2 = pts[(i + 1) % pts.length];
      newPts.push([p1[0] * 0.75 + p2[0] * 0.25, p1[1] * 0.75 + p2[1] * 0.25]);
      newPts.push([p1[0] * 0.25 + p2[0] * 0.75, p1[1] * 0.25 + p2[1] * 0.75]);
    }
    pts = newPts;
  }

  const segLens = pts.map((p, i) => {
    const q = pts[(i + 1) % pts.length];
    return Math.hypot(q[0] - p[0], q[1] - p[1]);
  });
  const totalLen = segLens.reduce((a, b) => a + b, 0);

  return { pts, segLens, totalLen, halfWidth: 22, N: pts.length };
}

// ── Track geometry helpers ────────────────────────────────────────────────────

function distToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-8) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function isOnTrack(x, y, track) {
  const { pts, halfWidth, N } = track;
  for (let i = 0; i < N; i++) {
    const a = pts[i], b = pts[(i + 1) % N];
    if (distToSeg(x, y, a[0], a[1], b[0], b[1]) <= halfWidth) return true;
  }
  return false;
}

function castRay(cx, cy, angle, track) {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  for (let d = RAY_STEP; d <= RAY_MAX; d += RAY_STEP) {
    if (!isOnTrack(cx + dx * d, cy + dy * d, track)) return (d - RAY_STEP) / RAY_MAX;
  }
  return 1.0;
}

function nearestSegIdx(x, y, track) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < track.N; i++) {
    const d = Math.hypot(x - track.pts[i][0], y - track.pts[i][1]);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// ── Car helpers ───────────────────────────────────────────────────────────────

function spawnCar(track) {
  const p0 = track.pts[0], p1 = track.pts[1];
  const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
  return {
    x: p0[0], y: p0[1],
    angle, speed: 10,
    alive: true,
    segIdx: 0,
    laps: 0,
    segProgress: 0,
    frames: 0,
    totalDist: 0,
  };
}

function senseCar(car, track) {
  const sensors = RAY_ANGLES.map(da => castRay(car.x, car.y, car.angle + da, track));
  sensors.push(car.speed / MAX_SPEED);
  sensors.push(car.angle / (2 * Math.PI));
  return new Float32Array(sensors);  // length INPUT_DIM = 11
}

function stepCar(car, steerOut, throttleOut) {
  // Grip loss bleeds speed when cornering hard at high velocity
  const gripLoss = GRIP_LOSS * Math.abs(steerOut) * (car.speed / MAX_SPEED);
  car.angle += steerOut * STEER_RATE * DT;
  const accel = throttleOut > 0 ?  throttleOut * ACCEL       * DT : 0;
  const brake = throttleOut < 0 ? -throttleOut * BRAKE_FORCE * DT : 0;
  car.speed = Math.max(0, Math.min(MAX_SPEED,
    (car.speed + accel - brake) * FRICTION * (1 - gripLoss)));
  car.x += Math.cos(car.angle) * car.speed * DT;
  car.y += Math.sin(car.angle) * car.speed * DT;
  car.frames++;
}

function updateCarProgress(car, track) {
  const N      = track.N;
  const newSeg = nearestSegIdx(car.x, car.y, track);
  let delta    = newSeg - car.segIdx;
  if (delta >  N / 2) delta -= N;
  if (delta < -N / 2) delta += N;
  if (delta > 0) {
    car.segProgress += delta;
    if (car.segProgress >= N) { car.laps++; car.segProgress -= N; }
    car.totalDist += delta;
  }
  car.segIdx = newSeg;
}

function carFitness(car) {
  const progress    = car.laps + car.totalDist / Math.max(1, car.frames) * (car.frames / _track.N);
  const speedBonus  = (car.frames > 0 ? car.speed / MAX_SPEED : 0) * 0.01;
  const deathFactor = car.alive ? 1 : 0.8;
  return (progress + speedBonus) * deathFactor;
}

// ── Neural net forward pass ───────────────────────────────────────────────────

function netForward(model, inputArr) {
  const x   = new T.Tensor([1, INPUT_DIM], inputArr, false);
  const out = model.forward(x, { training: false });
  return out.data;
}

// ── Box-Muller Gaussian noise ─────────────────────────────────────────────────

function gaussRand() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

// ── Population step ───────────────────────────────────────────────────────────

function stepGeneration(ticks) {
  if (!_pop || !_track || !_running) return null;
  ticks = Math.max(1, Math.min(ticks | 0, 6));

  for (let t = 0; t < ticks; t++) {
    let allDead = true;

    for (let i = 0; i < _popSize; i++) {
      const car = _cars[i];
      if (!car.alive) continue;
      allDead = false;

      const inputs   = senseCar(car, _track);
      const outs     = netForward(_pop.individuals[i], inputs);
      const steer    = Math.max(-1, Math.min(1, outs[0]));
      const throttle = Math.max(-1, Math.min(1, outs[1]));

      stepCar(car, steer, throttle);
      updateCarProgress(car, _track);

      if (!isOnTrack(car.x, car.y, _track) || car.frames >= MAX_FRAMES) {
        car.alive    = false;
        _carFit[i]   = carFitness(car);
      }
    }

    if (allDead) {
      _pop.evaluate((_, idx) => _carFit[idx]);
      const stats = _pop.evolve();
      _generation++;
      _genHistory.push({ gen: _generation, best: +_pop.bestFitness.toFixed(3), mean: +stats.mean.toFixed(3) });
      if (_genHistory.length > 100) _genHistory.shift();
      if (_pop.bestFitness > _bestFit) _bestFit = _pop.bestFitness;
      _genBestFit = stats.max;

      if (_maxGens > 0 && _generation >= _maxGens) _running = false;

      _cars   = Array.from({ length: _popSize }, () => spawnCar(_track));
      _carFit = new Array(_popSize).fill(0);
      break;
    }
  }

  return buildVisualState();
}

function buildVisualState() {
  if (!_cars || !_track) return null;
  return {
    track: _track,
    cars: _cars.map(c => ({
      x: c.x, y: c.y, angle: c.angle, alive: c.alive,
      speed: c.speed, laps: c.laps,
    })),
    generation: _generation,
    aliveCnt:   _cars.filter(c => c.alive).length,
    popSize:    _popSize,
    bestFit:    +_bestFit.toFixed(3),
    genBestFit: +_genBestFit.toFixed(3),
    genHistory: _genHistory.slice(-60),
    hasBestGenome: _pop !== null && _pop.bestIndividual !== null,
  };
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
module.exports = {
  mainHandlers: {
    'self-driving-car:init': (_, opts = {}) => {
      if (!Population || !T) return { error: 'Neuroevolution engine unavailable.' };

      _popSize = Math.max(4, Math.min(100, (opts.popSize | 0) || DEFAULT_POP_SIZE));
      _maxGens = Math.max(0, (opts.generations | 0) || 0);
      const seed   = (opts.seed || 0) >>> 0;
      const mutStd = Math.max(0.001, opts.mutStd || 0.05);

      _track = generateTrack(seed || 0xC0FFEE);

      _pop = new Population({
        architecture: {
          kind: 'classifier',
          inputDim: INPUT_DIM, outputDim: OUTPUT_DIM,
          hidden: [64, 32, 16], activation: 'tanh', dropout: 0,
        },
        size: _popSize,
        eliteCount: Math.max(1, Math.floor(_popSize * 0.2)),
        pMutate: 0.15, mutationStd: mutStd,
        tournamentK: 3, seed: opts.seed || 42,
      });

      _cars       = Array.from({ length: _popSize }, () => spawnCar(_track));
      _carFit     = new Array(_popSize).fill(0);
      _generation = 0;
      _bestFit    = 0;
      _genBestFit = 0;
      _running    = true;
      _genHistory = [];

      return { ok: true, track: _track };
    },

    'self-driving-car:getState': () => buildVisualState(),

    'self-driving-car:step': (_, ticks = 2) => stepGeneration(ticks),

    'self-driving-car:start': () => { _running = true;  return { ok: true }; },
    'self-driving-car:stop':  () => { _running = false; return { ok: true }; },

    'self-driving-car:newTrack': (_, seed) => {
      if (!Population) return { error: 'Not initialized.' };
      _track  = generateTrack((seed || Math.floor(Math.random() * 0xFFFFFF)) >>> 0);
      _cars   = Array.from({ length: _popSize }, () => spawnCar(_track));
      _carFit = new Array(_popSize).fill(0);
      return { ok: true, track: _track };
    },

    'self-driving-car:reset': () => {
      if (!Population || !T) return { error: 'Not initialized.' };
      const seed = Math.floor(Math.random() * 0xFFFFFF);
      _track = generateTrack(seed);
      _pop   = new Population({
        architecture: {
          kind: 'classifier',
          inputDim: INPUT_DIM, outputDim: OUTPUT_DIM,
          hidden: [64, 32, 16], activation: 'tanh', dropout: 0,
        },
        size: _popSize,
        eliteCount: Math.max(1, Math.floor(_popSize * 0.2)),
        pMutate: 0.15, mutationStd: 0.05,
        tournamentK: 3, seed: seed,
      });
      _cars       = Array.from({ length: _popSize }, () => spawnCar(_track));
      _carFit     = new Array(_popSize).fill(0);
      _generation = 0;
      _bestFit    = 0;
      _genBestFit = 0;
      _running    = true;
      _genHistory = [];
      return { ok: true };
    },

    // ── Inference handlers ──────────────────────────────────────────────────

    'self-driving-car:inferInit': () => {
      if (!_pop || !_track) return { error: 'No trained model — run training first.' };
      const genome = _pop.bestIndividual || _pop.individuals[0];
      if (!genome) return { error: 'Population not ready.' };
      _inferCar   = spawnCar(_track);
      _inferTrack = _track;
      return {
        ok: true,
        track: _inferTrack,
        generation: _generation,
        bestFit: +_bestFit.toFixed(3),
      };
    },

    'self-driving-car:inferStep': (_, noiseStd = 0) => {
      if (!_inferCar || !_inferTrack || !_pop) return null;
      const genome = _pop.bestIndividual || _pop.individuals[0];
      if (!genome) return null;

      const rawInputs = senseCar(_inferCar, _inferTrack);
      let inputs = rawInputs;
      if (noiseStd > 0) {
        inputs = new Float32Array(rawInputs.length);
        for (let i = 0; i < rawInputs.length; i++) {
          inputs[i] = rawInputs[i] + gaussRand() * noiseStd;
        }
      }

      const outs     = netForward(genome, inputs);
      const steer    = Math.max(-1, Math.min(1, outs[0]));
      const throttle = Math.max(-1, Math.min(1, outs[1]));
      stepCar(_inferCar, steer, throttle);
      updateCarProgress(_inferCar, _inferTrack);

      const dead = !isOnTrack(_inferCar.x, _inferCar.y, _inferTrack) || _inferCar.frames >= MAX_FRAMES;
      if (dead) _inferCar = spawnCar(_inferTrack);

      return {
        track: _inferTrack,
        car: { x: _inferCar.x, y: _inferCar.y, angle: _inferCar.angle, speed: _inferCar.speed, laps: _inferCar.laps },
        justReset: dead,
      };
    },
  },
};
