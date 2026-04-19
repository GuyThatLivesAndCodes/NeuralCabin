'use strict';

const { tokenize } = require('./lexer');
const { parse } = require('./parser');
const T = require('../engine/tensor');
const { buildFromSpec } = require('../engine/model');
const { buildOptim } = require('../engine/optim');
const { trainNetwork, trainNetworkSync, infer } = require('../engine/trainer');

class ReturnException { constructor(value) { this.value = value; } }

function truthy(v) {
  if (v === null || v === undefined || v === false || v === 0 || v === '') return false;
  if (Array.isArray(v) && v.length === 0) return false;
  return true;
}

function evalNode(node, env, ctx) {
  switch (node.type) {
    case 'Program': {
      let result = null;
      for (const s of node.body) result = evalNode(s, env, ctx);
      return result;
    }
    case 'Let': {
      env.define(node.name, evalNode(node.value, env, ctx));
      return null;
    }
    case 'Set': {
      const v = evalNode(node.value, env, ctx);
      if (!node.chain || node.chain.length === 0) {
        env.assign(node.name, v);
      } else {
        let target = env.get(node.name);
        for (let i = 0; i < node.chain.length - 1; i++) {
          const step = node.chain[i];
          const key = step.kind === 'dot' ? step.key : evalNode(step.key, env, ctx);
          target = target[key];
        }
        const last = node.chain[node.chain.length - 1];
        const key = last.kind === 'dot' ? last.key : evalNode(last.key, env, ctx);
        target[key] = v;
      }
      return null;
    }
    case 'If': {
      if (truthy(evalNode(node.cond, env, ctx))) {
        for (const s of node.consequent) evalNode(s, env, ctx);
      } else if (node.alternate) {
        for (const s of node.alternate) evalNode(s, env, ctx);
      }
      return null;
    }
    case 'While': {
      let guard = 0;
      while (truthy(evalNode(node.cond, env, ctx))) {
        for (const s of node.body) evalNode(s, env, ctx);
        if (++guard > 10_000_000) throw new Error('while: iteration limit exceeded');
      }
      return null;
    }
    case 'For': {
      let from = evalNode(node.from, env, ctx);
      const to = evalNode(node.to, env, ctx);
      const by = node.by ? evalNode(node.by, env, ctx) : 1;
      const forEnv = new Env(env);
      forEnv.define(node.name, from);
      let guard = 0;
      while ((by >= 0 ? forEnv.get(node.name) <= to : forEnv.get(node.name) >= to)) {
        for (const s of node.body) evalNode(s, forEnv, ctx);
        forEnv.assign(node.name, forEnv.get(node.name) + by);
        if (++guard > 10_000_000) throw new Error('for: iteration limit exceeded');
      }
      return null;
    }
    case 'FnDecl': {
      const fn = makeFunction(node.params, node.body, env);
      env.define(node.name, fn);
      return null;
    }
    case 'Return': {
      throw new ReturnException(node.value ? evalNode(node.value, env, ctx) : null);
    }
    case 'Print': {
      const v = evalNode(node.value, env, ctx);
      ctx.output.push(stringify(v));
      return null;
    }
    case 'ExpressionStatement': return evalNode(node.expression, env, ctx);
    case 'Num': return node.value;
    case 'Str': return node.value;
    case 'Literal': return node.value;
    case 'Ident': return env.get(node.name);
    case 'List': return node.items.map(it => evalNode(it, env, ctx));
    case 'Obj': {
      const o = {};
      for (const e of node.entries) o[e.key] = evalNode(e.value, env, ctx);
      return o;
    }
    case 'Unary': {
      const v = evalNode(node.value, env, ctx);
      if (node.op === '-') return -v;
      if (node.op === 'not') return !truthy(v);
      throw new Error('bad unary ' + node.op);
    }
    case 'Logical': {
      const l = evalNode(node.left, env, ctx);
      if (node.op === 'and') return truthy(l) ? evalNode(node.right, env, ctx) : l;
      return truthy(l) ? l : evalNode(node.right, env, ctx);
    }
    case 'Binary': {
      const l = evalNode(node.left, env, ctx);
      const r = evalNode(node.right, env, ctx);
      switch (node.op) {
        case '+': return (typeof l === 'string' || typeof r === 'string') ? String(l) + String(r) : l + r;
        case '-': return l - r;
        case '*': return l * r;
        case '/': return l / r;
        case '%': return l % r;
        case '==': return l === r;
        case '!=': return l !== r;
        case '<': return l < r;
        case '<=': return l <= r;
        case '>': return l > r;
        case '>=': return l >= r;
      }
      throw new Error('bad binary ' + node.op);
    }
    case 'Call': {
      const fn = evalNode(node.callee, env, ctx);
      if (typeof fn !== 'function') throw new Error(`cannot call: ${stringify(fn)}`);
      const args = node.args.map(a => evalNode(a, env, ctx));
      return fn(...args);
    }
    case 'Index': {
      const obj = evalNode(node.object, env, ctx);
      const key = evalNode(node.key, env, ctx);
      if (obj == null) throw new Error('indexing null');
      return obj[key];
    }
    case 'Dot': {
      const obj = evalNode(node.object, env, ctx);
      if (obj == null) throw new Error('property of null: ' + node.name);
      return obj[node.name];
    }
  }
  throw new Error('unknown node ' + node.type);
}

function makeFunction(params, body, closure) {
  return function (...args) {
    const env = new Env(closure);
    for (let i = 0; i < params.length; i++) env.define(params[i], args[i]);
    try {
      for (const s of body) evalNode(s, env, env._ctx);
      return null;
    } catch (e) {
      if (e instanceof ReturnException) return e.value;
      throw e;
    }
  };
}

class Env {
  constructor(parent) {
    this.parent = parent || null;
    this.vars = new Map();
    this._ctx = parent ? parent._ctx : null;
  }
  define(name, value) { this.vars.set(name, value); }
  get(name) {
    if (this.vars.has(name)) return this.vars.get(name);
    if (this.parent) return this.parent.get(name);
    throw new Error(`undefined: ${name}`);
  }
  assign(name, value) {
    if (this.vars.has(name)) { this.vars.set(name, value); return; }
    if (this.parent && this.parent._has(name)) { this.parent.assign(name, value); return; }
    this.vars.set(name, value); // implicit define if nowhere
  }
  _has(name) { return this.vars.has(name) || (this.parent && this.parent._has(name)); }
}

function stringify(v) {
  if (v == null) return String(v);
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return '[' + v.map(stringify).join(', ') + ']';
  if (typeof v === 'function') return '<fn>';
  try { return JSON.stringify(v); } catch { return '<object>'; }
}

function makeStandardLib(scriptCtx) {
  const lib = {
    len: (x) => (x == null ? 0 : x.length),
    push: (arr, v) => { arr.push(v); return arr; },
    range: (a, b, step) => {
      const out = [];
      if (b === undefined) { for (let i = 0; i < a; i++) out.push(i); return out; }
      const s = step ?? 1;
      if (s > 0) for (let i = a; i < b; i += s) out.push(i);
      else for (let i = a; i > b; i += s) out.push(i);
      return out;
    },
    str: stringify,
    num: (s) => Number(s),
    abs: Math.abs, min: Math.min, max: Math.max,
    sqrt: Math.sqrt, exp: Math.exp, log: Math.log,
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    floor: Math.floor, ceil: Math.ceil, round: Math.round,
    random: Math.random,
    keys: (o) => Object.keys(o),
    values: (o) => Object.values(o),
    // Neural network API
    build: (spec) => buildFromSpec(spec),
    tensor: (shape, data) => new T.Tensor(shape, data, false),
    forward: (model, x) => model.forward(x, { training: false }),
    train: (spec, trainingData, trainingOpts) => {
      const net = {
        architecture: spec,
        training: trainingOpts || { optimizer: 'adam', learningRate: 0.01, batchSize: 16, epochs: 10, seed: 42 },
        trainingData,
        state: null
      };
      return trainNetworkSync(net, {
        onProgress: (p) => scriptCtx.output.push(`[train] ep ${p.epoch}/${p.totalEpochs} step ${p.step}/${p.totalSteps} loss=${p.loss.toFixed(4)}`),
        shouldStop: () => false,
        log: (l) => scriptCtx.output.push('[train] ' + l)
      });
    },
    predict: (network, input) => infer(network, input),
    thisNet: () => scriptCtx.network
  };
  return lib;
}

async function runScript(code, ctx = {}) {
  const tokens = tokenize(code);
  const ast = parse(tokens);
  const scriptCtx = { output: [], network: ctx.network || null, saveTrainedState: ctx.saveTrainedState };
  const globalEnv = new Env(null);
  globalEnv._ctx = scriptCtx;
  const lib = makeStandardLib(scriptCtx);
  for (const [k, v] of Object.entries(lib)) globalEnv.define(k, v);

  try {
    // Execute top-level. Some stdlib functions are async (train); since our
    // interpreter is synchronous, async funcs return a Promise the script
    // can handle via `let r = await(train(...))`. Implement `await(x)` helper.
    globalEnv.define('await', (p) => p); // pass-through — train is sync in NeuralScript
    evalNode(ast, globalEnv, scriptCtx);
    return { ok: true, output: scriptCtx.output.join('\n') };
  } catch (e) {
    return { ok: false, error: e.message, output: scriptCtx.output.join('\n') };
  }
}

module.exports = { runScript };
