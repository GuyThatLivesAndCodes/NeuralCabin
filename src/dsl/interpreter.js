'use strict';

const { compileNeuralScript } = require('./compiler');
const T = require('../engine/tensor');
const { buildFromSpec } = require('../engine/model');
const { trainNetwork, trainNetworkSync, infer } = require('../engine/trainer');

// 1. SIGNAL CLASS: Replacing exceptions for control flow.
// Throwing/catching exceptions is incredibly slow because it builds stack traces.
class ReturnSignal { 
  constructor(value) { 
    this.value = value; 
  } 
}

// 2. FAST TRUTHINESS: Avoid unnecessary function calls for common falsy values.
function truthy(v) {
  if (!v) return false; // Instantly catches null, undefined, false, 0, ''
  if (Array.isArray(v) && v.length === 0) return false;
  return true;
}

function evalNode(node, env, ctx) {
  switch (node.type) {
    case 'Program': {
      let result = null;
      // Using standard loops instead of for...of avoids iterator allocation overhead
      const len = node.body.length;
      for (let i = 0; i < len; i++) {
        result = evalNode(node.body[i], env, ctx);
        if (result instanceof ReturnSignal) return result;
      }
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
        const chainLen = node.chain.length;
        for (let i = 0; i < chainLen - 1; i++) {
          const step = node.chain[i];
          const key = step.kind === 'dot' ? step.key : evalNode(step.key, env, ctx);
          target = target[key];
        }
        const last = node.chain[chainLen - 1];
        const key = last.kind === 'dot' ? last.key : evalNode(last.key, env, ctx);
        target[key] = v;
      }
      return null;
    }
    case 'If': {
      if (truthy(evalNode(node.cond, env, ctx))) {
        const len = node.consequent.length;
        for (let i = 0; i < len; i++) {
          const res = evalNode(node.consequent[i], env, ctx);
          if (res instanceof ReturnSignal) return res; // Bubble up return signal
        }
      } else if (node.alternate) {
        const len = node.alternate.length;
        for (let i = 0; i < len; i++) {
          const res = evalNode(node.alternate[i], env, ctx);
          if (res instanceof ReturnSignal) return res;
        }
      }
      return null;
    }
    case 'While': {
      let guard = 0;
      while (truthy(evalNode(node.cond, env, ctx))) {
        const len = node.body.length;
        for (let i = 0; i < len; i++) {
          const res = evalNode(node.body[i], env, ctx);
          if (res instanceof ReturnSignal) return res;
        }
        if (++guard > 10_000_000) throw new Error('while: iteration limit exceeded');
      }
      return null;
    }
    case 'For': {
      let from = evalNode(node.from, env, ctx);
      const to = evalNode(node.to, env, ctx);
      const by = node.by ? evalNode(node.by, env, ctx) : 1;
      
      const forEnv = new Env(env);
      // Fast path: bypass .define() overhead and inject directly
      forEnv.vars[node.name] = from; 
      
      let guard = 0;
      const isUp = by >= 0;
      const len = node.body.length;

      while (true) {
        // Fast path: direct dictionary access, avoiding prototype/scope traversal
        const current = forEnv.vars[node.name]; 
        if (isUp ? current > to : current < to) break;

        for (let i = 0; i < len; i++) {
          const res = evalNode(node.body[i], forEnv, ctx);
          if (res instanceof ReturnSignal) return res;
        }
        
        forEnv.vars[node.name] += by; // Fast path: Direct local scope mutation
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
      // Create lightweight signal object instead of throwing
      return new ReturnSignal(node.value ? evalNode(node.value, env, ctx) : null);
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
    case 'List': {
      const len = node.items.length;
      const arr = new Array(len); // Pre-allocate array
      for (let i = 0; i < len; i++) {
        arr[i] = evalNode(node.items[i], env, ctx);
      }
      return arr;
    }
    case 'Obj': {
      const o = {};
      const len = node.entries.length;
      for (let i = 0; i < len; i++) {
        const e = node.entries[i];
        o[e.key] = evalNode(e.value, env, ctx);
      }
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
      
      const len = node.args.length;
      const args = new Array(len); // Pre-allocate for speed
      for (let i = 0; i < len; i++) {
        args[i] = evalNode(node.args[i], env, ctx);
      }
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
    const paramsLen = params.length;
    // Direct assignment bypasses .define() checks
    for (let i = 0; i < paramsLen; i++) env.vars[params[i]] = args[i]; 
    
    const bodyLen = body.length;
    for (let i = 0; i < bodyLen; i++) {
      const res = evalNode(body[i], env, env._ctx);
      if (res instanceof ReturnSignal) return res.value; // Unpack signal here
    }
    return null;
  };
}

class Env {
  constructor(parent) {
    this.parent = parent || null;
    // Object.create(null) avoids prototype chain traversals and is highly 
    // optimized by V8 engines for fast dictionary/hash-map lookups
    this.vars = Object.create(null);
    this._ctx = parent ? parent._ctx : null;
  }
  
  define(name, value) { 
    this.vars[name] = value; 
  }
  
  get(name) {
    let current = this;
    while (current !== null) {
      if (name in current.vars) return current.vars[name];
      current = current.parent;
    }
    throw new Error(`undefined: ${name}`);
  }
  
  assign(name, value) {
    let current = this;
    while (current !== null) {
      if (name in current.vars) { 
        current.vars[name] = value; 
        return; 
      }
      current = current.parent;
    }
    this.vars[name] = value; // implicit define if nowhere
  }
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
  const scriptCtx = { output: [], network: ctx.network || null, saveTrainedState: ctx.saveTrainedState };
  
  const globalEnv = new Env(null);
  globalEnv._ctx = scriptCtx;
  
  const lib = makeStandardLib(scriptCtx);
  for (const [k, v] of Object.entries(lib)) globalEnv.define(k, v);

  try {
    const { ast } = compileNeuralScript(code);
    globalEnv.define('await', (p) => p); 
    
    const result = evalNode(ast, globalEnv, scriptCtx);
    // Note: If a top-level return is executed, result will be a ReturnSignal, which we can safely ignore
    
    return { ok: true, output: scriptCtx.output.join('\n') };
  } catch (e) {
    return { ok: false, error: e.message, output: scriptCtx.output.join('\n') };
  }
}

module.exports = { runScript };
