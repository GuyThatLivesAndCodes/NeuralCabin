'use strict';

const path = require('path');
const { app } = require('electron');

let _DQNAgent;
try {
  _DQNAgent = require(path.join(app.getAppPath(), 'src', 'engine', 'rl')).DQNAgent;
} catch (e) {
  console.error('[warehouse-robot] Failed to load DQNAgent:', e.message);
}

// ── Environment constants ─────────────────────────────────────────────────────
const GRID      = 8;
const N_BOXES   = 3;
const MAX_STEPS = 200;
const STATE_DIM = 2 + N_BOXES * 2 + N_BOXES * 2;  // 14 floats
const N_ACTIONS = 4;
const DIRS      = [[-1, 0], [1, 0], [0, -1], [0, 1]];

// ── Per-instance sessions ─────────────────────────────────────────────────────
// Each network gets its own isolated state keyed by instanceId.
function makeSession() {
  return {
    agent: null, env: null, running: false,
    episode: 0, epReward: 0, totalSteps: 0,
    bestReward: -Infinity, rewardHistory: [],
    inferEnv: null, inferEpReward: 0, inferLapsDone: 0,
  };
}
const _sessions = new Map();
function getSession(id) {
  const key = id || 'default';
  if (!_sessions.has(key)) _sessions.set(key, makeSession());
  return _sessions.get(key);
}

// ── LCG for seeded random positions ──────────────────────────────────────────
function lcg(s) { return (Math.imul(s | 0, 1664525) + 1013904223) >>> 0; }

// ── Grid helpers ──────────────────────────────────────────────────────────────

function resetEnv(seed) {
  let rng = (seed || (Math.floor(Math.random() * 0x7FFFFFFF) + 1)) >>> 0;
  const used = new Set();

  function randCell() {
    let r, c;
    do {
      rng = lcg(rng); r = rng % GRID;
      rng = lcg(rng); c = rng % GRID;
    } while (used.has(r * GRID + c));
    used.add(r * GRID + c);
    return [r, c];
  }

  const robot   = randCell();
  const boxes   = [randCell(), randCell(), randCell()];
  const targets = [randCell(), randCell(), randCell()];
  return { robot, boxes, targets, step: 0, onTarget: countOnTarget(boxes, targets) };
}

function countOnTarget(boxes, targets) {
  let n = 0;
  for (const t of targets)
    if (boxes.some(b => b[0] === t[0] && b[1] === t[1])) n++;
  return n;
}

function encodeState(env) {
  const G = GRID - 1;
  const v = new Float32Array(STATE_DIM);
  v[0] = env.robot[0] / G;
  v[1] = env.robot[1] / G;
  for (let i = 0; i < N_BOXES; i++) {
    v[2 + i * 2]     = env.boxes[i][0] / G;
    v[2 + i * 2 + 1] = env.boxes[i][1] / G;
  }
  for (let i = 0; i < N_BOXES; i++) {
    v[2 + N_BOXES * 2 + i * 2]     = env.targets[i][0] / G;
    v[2 + N_BOXES * 2 + i * 2 + 1] = env.targets[i][1] / G;
  }
  return v;
}

function stepEnv(env, action) {
  const [dr, dc] = DIRS[action];
  const [r, c]   = env.robot;
  const nr = r + dr, nc = c + dc;

  if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) {
    return { env: { ...env, step: env.step + 1 }, reward: -0.5, done: env.step + 1 >= MAX_STEPS };
  }

  let boxes = env.boxes;
  const bi  = boxes.findIndex(b => b[0] === nr && b[1] === nc);

  if (bi >= 0) {
    const bnr = nr + dr, bnc = nc + dc;
    if (bnr < 0 || bnr >= GRID || bnc < 0 || bnc >= GRID ||
        boxes.some((b, i) => i !== bi && b[0] === bnr && b[1] === bnc)) {
      return { env: { ...env, step: env.step + 1 }, reward: -0.4, done: env.step + 1 >= MAX_STEPS };
    }
    boxes = boxes.map((b, i) => i === bi ? [bnr, bnc] : b);
  }

  const newEnv = { ...env, robot: [nr, nc], boxes, step: env.step + 1 };
  const onNow  = countOnTarget(newEnv.boxes, newEnv.targets);
  const onPrev = env.onTarget;
  newEnv.onTarget = onNow;

  let reward = -0.01;
  if (onNow > onPrev) reward += 10 * (onNow - onPrev);
  if (onNow < onPrev) reward -= 3  * (onPrev - onNow);

  const done = (onNow === N_BOXES) || (newEnv.step >= MAX_STEPS);
  if (onNow === N_BOXES) reward += 50;

  return { env: newEnv, reward, done };
}

function buildVisualState(s) {
  if (!s.env) return null;
  return {
    grid: GRID, nBoxes: N_BOXES,
    robot:    s.env.robot,
    boxes:    s.env.boxes,
    targets:  s.env.targets,
    onTarget: s.env.onTarget,
    episode:  s.episode,
    stepInEp: s.env.step,
    totalSteps: s.totalSteps,
    epReward: +s.epReward.toFixed(2),
    bestReward: s.bestReward === -Infinity ? null : +s.bestReward.toFixed(2),
    epsilon:  s.agent ? +s.agent.epsilon.toFixed(4) : 1.0,
    rewardHistory: s.rewardHistory.slice(-80),
  };
}

// ── Gaussian noise ────────────────────────────────────────────────────────────
function gaussRand() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
module.exports = {
  mainHandlers: {
    'warehouse-robot:init': (_, opts = {}) => {
      if (!_DQNAgent) return { error: 'DQNAgent module unavailable — check app bundle.' };
      const id = opts.instanceId || 'default';
      const s  = getSession(id);
      s.agent = new _DQNAgent({
        architecture: {
          kind: 'classifier',
          inputDim: STATE_DIM, outputDim: N_ACTIONS,
          hidden: [128, 64], activation: 'relu', dropout: 0,
        },
        gamma: 0.95,
        lr: opts.lr || opts.learningRate || 1e-3,
        batchSize:      opts.batchSize || 64,
        bufferCapacity: 5000,   // smaller buffer = less NAPI data per sample call
        epsilonStart: 1.0, epsilonEnd: 0.05, epsilonDecay: 0.9995,
        targetUpdateFreq: 200,
        seed: opts.seed || 42,
        optimizer: 'adam',
      });
      s.env          = resetEnv(opts.seed);
      s.running      = true;
      s.episode      = 0;
      s.epReward     = 0;
      s.totalSteps   = 0;
      s.bestReward   = -Infinity;
      s.rewardHistory = [];
      return { ok: true, grid: GRID, nBoxes: N_BOXES };
    },

    'warehouse-robot:getState': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      return buildVisualState(getSession(id));
    },

    // KEY FIX: collect n steps, then train ONCE — avoids blocking the main
    // process with N full backprop passes per IPC call.
    'warehouse-robot:step': (_, opts = {}) => {
      const id = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
      const s  = getSession(id);
      if (!s.agent || !s.env || !s.running) return buildVisualState(s);
      const n = Math.max(1, Math.min((opts.n || (typeof opts === 'number' ? opts : 4)) | 0, 20));

      for (let i = 0; i < n; i++) {
        const state  = encodeState(s.env);
        const a      = s.agent.selectAction(state);
        const { env: next, reward, done } = stepEnv(s.env, a);
        const ns     = encodeState(next);
        s.agent.observe(state, a, reward, ns, done);
        s.epReward   += reward;
        s.totalSteps += 1;
        s.env         = next;

        if (done) {
          s.rewardHistory.push(+s.epReward.toFixed(2));
          if (s.rewardHistory.length > 200) s.rewardHistory.shift();
          if (s.epReward > s.bestReward) s.bestReward = s.epReward;
          s.episode++;
          s.epReward = 0;
          s.env = resetEnv();
        }
      }

      // Train once per IPC call rather than once per step.
      // This prevents the replay buffer's NAPI array conversion from
      // blocking the main process N times per frame.
      if (s.agent.buffer.ready) s.agent.trainStep();

      return buildVisualState(s);
    },

    'warehouse-robot:start': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      getSession(id).running = true;
      return { ok: true };
    },
    'warehouse-robot:stop': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      getSession(id).running = false;
      return { ok: true };
    },

    'warehouse-robot:reset': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      const s  = getSession(id);
      if (!s.agent) return { error: 'Not initialized — call init first.' };
      s.env          = resetEnv();
      s.episode      = 0;
      s.epReward     = 0;
      s.totalSteps   = 0;
      s.bestReward   = -Infinity;
      s.rewardHistory = [];
      s.agent.epsilon = s.agent.epsilonStart;
      s.running      = true;
      return { ok: true };
    },

    // ── Inference handlers ────────────────────────────────────────────────

    'warehouse-robot:inferInit': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      const s  = getSession(id);
      if (!s.agent) return { error: 'No trained agent — run training first.' };
      s.inferEnv      = resetEnv();
      s.inferEpReward = 0;
      s.inferLapsDone = 0;
      return {
        ok: true, grid: GRID, nBoxes: N_BOXES,
        epsilon:    +s.agent.epsilon.toFixed(4),
        totalSteps: s.totalSteps,
        episode:    s.episode,
        bestReward: s.bestReward === -Infinity ? null : +s.bestReward.toFixed(2),
      };
    },

    'warehouse-robot:inferStep': (_, opts = {}) => {
      const id       = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
      const noiseStd = (typeof opts === 'object' ? opts.noiseStd : 0) || 0;
      const s        = getSession(id);
      if (!s.agent || !s.inferEnv) return null;

      const rawState = encodeState(s.inferEnv);
      let state = rawState;
      if (noiseStd > 0) {
        state = new Float32Array(rawState.length);
        for (let i = 0; i < rawState.length; i++) state[i] = rawState[i] + gaussRand() * noiseStd;
      }

      const savedEps = s.agent.epsilon;
      s.agent.epsilon = 0;
      const action = s.agent.selectAction(state);
      s.agent.epsilon = savedEps;

      const { env: next, reward, done } = stepEnv(s.inferEnv, action);
      s.inferEpReward += reward;
      s.inferEnv       = next;

      let justReset = false;
      if (done) {
        s.inferLapsDone++;
        s.inferEpReward = 0;
        s.inferEnv      = resetEnv();
        justReset       = true;
      }

      return {
        grid: GRID, nBoxes: N_BOXES,
        robot:    s.inferEnv.robot,
        boxes:    s.inferEnv.boxes,
        targets:  s.inferEnv.targets,
        onTarget: s.inferEnv.onTarget,
        epReward: +s.inferEpReward.toFixed(2),
        episodesDone: s.inferLapsDone,
        justReset,
      };
    },
  },
};
