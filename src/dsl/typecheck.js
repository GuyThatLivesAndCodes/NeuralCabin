'use strict';

const TypeKind = {
  ANY: 'any',
  NUMBER: 'number',
  STRING: 'string',
  BOOLEAN: 'boolean',
  NULL: 'null',
  LIST: 'list',
  OBJECT: 'object',
  FUNCTION: 'function',
  NETWORK: 'network',
  TENSOR: 'tensor'
};

class Scope {
  constructor(parent = null) {
    this.parent = parent;
    this.bindings = Object.create(null);
  }
  define(name, info) {
    this.bindings[name] = info;
  }
  assign(name, info) {
    let cur = this;
    while (cur) {
      if (Object.prototype.hasOwnProperty.call(cur.bindings, name)) {
        cur.bindings[name] = { ...cur.bindings[name], ...info };
        return;
      }
      cur = cur.parent;
    }
    this.bindings[name] = info;
  }
  get(name) {
    let cur = this;
    while (cur) {
      if (Object.prototype.hasOwnProperty.call(cur.bindings, name)) return cur.bindings[name];
      cur = cur.parent;
    }
    return null;
  }
}

function err(errors, line, message) {
  errors.push({ line: line || 0, message });
}

function isPositiveInt(v) {
  return Number.isInteger(v) && v > 0;
}

function isNonNegativeInt(v) {
  return Number.isInteger(v) && v >= 0;
}

function inferLiteral(node) {
  if (!node) return { ok: false };
  switch (node.type) {
    case 'Num':
      return { ok: true, value: node.value };
    case 'Str':
      return { ok: true, value: node.value };
    case 'Literal':
      return { ok: true, value: node.value };
    default:
      return { ok: false };
  }
}

function inferStatic(node, scope, lineHint) {
  if (!node) return { ok: false };
  const lit = inferLiteral(node);
  if (lit.ok) return lit;

  if (node.type === 'Ident') {
    const b = scope.get(node.name);
    if (b && b.hasConst) return { ok: true, value: b.constValue };
    return { ok: false };
  }

  if (node.type === 'List') {
    const out = [];
    for (const item of node.items) {
      const s = inferStatic(item, scope, lineHint);
      if (!s.ok) return { ok: false };
      out.push(s.value);
    }
    return { ok: true, value: out };
  }

  if (node.type === 'Obj') {
    const out = {};
    for (const entry of node.entries) {
      const s = inferStatic(entry.value, scope, lineHint);
      if (!s.ok) return { ok: false };
      out[entry.key] = s.value;
    }
    return { ok: true, value: out };
  }

  if (node.type === 'Unary') {
    const s = inferStatic(node.value, scope, lineHint);
    if (!s.ok) return { ok: false };
    if (node.op === '-') return { ok: true, value: -s.value };
    if (node.op === 'not') return { ok: true, value: !s.value };
    return { ok: false };
  }

  if (node.type === 'Binary') {
    const l = inferStatic(node.left, scope, lineHint);
    const r = inferStatic(node.right, scope, lineHint);
    if (!l.ok || !r.ok) return { ok: false };
    switch (node.op) {
      case '+': return { ok: true, value: l.value + r.value };
      case '-': return { ok: true, value: l.value - r.value };
      case '*': return { ok: true, value: l.value * r.value };
      case '/': return { ok: true, value: l.value / r.value };
      case '%': return { ok: true, value: l.value % r.value };
      default: return { ok: false };
    }
  }

  return { ok: false };
}

function inferType(node, scope, errors, lineHint) {
  if (!node) return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };

  switch (node.type) {
    case 'Num':
      return { kind: TypeKind.NUMBER, hasConst: true, constValue: node.value };
    case 'Str':
      return { kind: TypeKind.STRING, hasConst: true, constValue: node.value };
    case 'Literal': {
      if (node.value === null) return { kind: TypeKind.NULL, hasConst: true, constValue: null };
      if (typeof node.value === 'boolean') return { kind: TypeKind.BOOLEAN, hasConst: true, constValue: node.value };
      return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
    }
    case 'Ident': {
      const b = scope.get(node.name);
      if (!b) return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
      return b;
    }
    case 'List': {
      const values = [];
      let allConst = true;
      for (const item of node.items) {
        const t = inferType(item, scope, errors, lineHint);
        if (!t.hasConst) allConst = false;
        values.push(t.constValue);
      }
      return {
        kind: TypeKind.LIST,
        hasConst: allConst,
        constValue: allConst ? values : undefined
      };
    }
    case 'Obj': {
      const out = {};
      let allConst = true;
      for (const entry of node.entries) {
        const t = inferType(entry.value, scope, errors, lineHint);
        if (!t.hasConst) allConst = false;
        out[entry.key] = t.constValue;
      }
      return {
        kind: TypeKind.OBJECT,
        hasConst: allConst,
        constValue: allConst ? out : undefined
      };
    }
    case 'Unary': {
      const t = inferType(node.value, scope, errors, lineHint);
      if (node.op === '-' && t.kind !== TypeKind.NUMBER && t.kind !== TypeKind.ANY) {
        err(errors, lineHint, `Unary '-' expects number, got ${t.kind}`);
      }
      if (node.op === 'not' && t.kind !== TypeKind.BOOLEAN && t.kind !== TypeKind.ANY) {
        err(errors, lineHint, `Unary 'not' expects boolean, got ${t.kind}`);
      }
      const s = inferStatic(node, scope, lineHint);
      return {
        kind: node.op === '-' ? TypeKind.NUMBER : TypeKind.BOOLEAN,
        hasConst: s.ok,
        constValue: s.ok ? s.value : undefined
      };
    }
    case 'Binary': {
      const l = inferType(node.left, scope, errors, lineHint);
      const r = inferType(node.right, scope, errors, lineHint);
      if (['-', '*', '/', '%'].includes(node.op)) {
        if (l.kind !== TypeKind.NUMBER && l.kind !== TypeKind.ANY) err(errors, lineHint, `Operator '${node.op}' expects numbers.`);
        if (r.kind !== TypeKind.NUMBER && r.kind !== TypeKind.ANY) err(errors, lineHint, `Operator '${node.op}' expects numbers.`);
        const s = inferStatic(node, scope, lineHint);
        return { kind: TypeKind.NUMBER, hasConst: s.ok, constValue: s.ok ? s.value : undefined };
      }
      if (node.op === '+') {
        const isNum = l.kind === TypeKind.NUMBER || l.kind === TypeKind.ANY;
        const isStr = l.kind === TypeKind.STRING || l.kind === TypeKind.ANY;
        const rNum = r.kind === TypeKind.NUMBER || r.kind === TypeKind.ANY;
        const rStr = r.kind === TypeKind.STRING || r.kind === TypeKind.ANY;
        if (!(isNum && rNum) && !(isStr || rStr)) {
          err(errors, lineHint, "Operator '+' expects number+number or string concatenation.");
        }
        const s = inferStatic(node, scope, lineHint);
        return {
          kind: (l.kind === TypeKind.STRING || r.kind === TypeKind.STRING) ? TypeKind.STRING : TypeKind.ANY,
          hasConst: s.ok,
          constValue: s.ok ? s.value : undefined
        };
      }
      if (['==', '!=', '<', '<=', '>', '>='].includes(node.op)) {
        return { kind: TypeKind.BOOLEAN, hasConst: false, constValue: undefined };
      }
      return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
    }
    case 'Logical':
      inferType(node.left, scope, errors, lineHint);
      inferType(node.right, scope, errors, lineHint);
      return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
    case 'Call':
      return inferCall(node, scope, errors, lineHint);
    case 'Index':
      inferType(node.object, scope, errors, lineHint);
      inferType(node.key, scope, errors, lineHint);
      return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
    case 'Dot':
      inferType(node.object, scope, errors, lineHint);
      return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
    default:
      return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
  }
}

function validateHidden(hidden, line, errors) {
  if (hidden === undefined) return;
  if (!Array.isArray(hidden)) {
    err(errors, line, 'architecture.hidden must be an array of positive integers.');
    return;
  }
  for (const v of hidden) {
    if (!isPositiveInt(v)) {
      err(errors, line, `architecture.hidden contains invalid width ${JSON.stringify(v)}.`);
      return;
    }
  }
}

function validateArchitectureSpec(spec, line, errors) {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
    err(errors, line, 'build/train requires a static architecture object.');
    return;
  }
  const kind = spec.kind;
  if (typeof kind !== 'string') {
    err(errors, line, "architecture.kind must be a string literal.");
    return;
  }

  if (!['classifier', 'mlp', 'regressor', 'charLM', 'gpt'].includes(kind)) {
    err(errors, line, `Unknown architecture.kind "${kind}".`);
    return;
  }

  if (kind === 'classifier' || kind === 'mlp' || kind === 'regressor') {
    if (!isPositiveInt(spec.inputDim)) err(errors, line, 'architecture.inputDim must be a positive integer.');
    if (!isPositiveInt(spec.outputDim)) err(errors, line, 'architecture.outputDim must be a positive integer.');
    validateHidden(spec.hidden, line, errors);
  }

  if (kind === 'classifier' && spec.classes !== undefined) {
    if (!Array.isArray(spec.classes) || spec.classes.some(v => typeof v !== 'string')) {
      err(errors, line, 'architecture.classes must be an array of strings.');
    } else if (isPositiveInt(spec.outputDim) && spec.classes.length && spec.classes.length !== spec.outputDim) {
      err(errors, line, `architecture.classes length (${spec.classes.length}) must match outputDim (${spec.outputDim}).`);
    }
  }

  if (kind === 'regressor' && spec.classes !== undefined) {
    err(errors, line, 'architecture.classes is not valid for regressor models.');
  }

  if (kind === 'charLM' || kind === 'gpt') {
    if (!isPositiveInt(spec.contextLen)) err(errors, line, 'architecture.contextLen must be a positive integer.');
    if (!isPositiveInt(spec.embDim)) err(errors, line, 'architecture.embDim must be a positive integer.');
    if (spec.vocabSize !== undefined && !isNonNegativeInt(spec.vocabSize)) {
      err(errors, line, 'architecture.vocabSize must be a non-negative integer.');
    }
    validateHidden(spec.hidden, line, errors);
  }
}

function validateTrainingOptions(opts, line, errors) {
  if (opts === undefined) return;
  if (!opts || typeof opts !== 'object' || Array.isArray(opts)) {
    err(errors, line, 'training options must be an object.');
    return;
  }
  if (opts.epochs !== undefined && !isPositiveInt(opts.epochs)) {
    err(errors, line, 'training.epochs must be a positive integer.');
  }
  if (opts.batchSize !== undefined && !isPositiveInt(opts.batchSize)) {
    err(errors, line, 'training.batchSize must be a positive integer.');
  }
  if (opts.learningRate !== undefined && !(typeof opts.learningRate === 'number' && Number.isFinite(opts.learningRate) && opts.learningRate > 0)) {
    err(errors, line, 'training.learningRate must be a positive number.');
  }
}

function inferCall(node, scope, errors, lineHint) {
  const calleeIsIdent = node.callee && node.callee.type === 'Ident';
  const calleeName = calleeIsIdent ? node.callee.name : null;
  const argTypes = node.args.map(arg => inferType(arg, scope, errors, lineHint));

  if (calleeName === 'build') {
    const arg0 = argTypes[0];
    if (!arg0) {
      err(errors, lineHint, 'build(spec) requires an architecture argument.');
    } else if (arg0.hasConst) {
      validateArchitectureSpec(arg0.constValue, lineHint, errors);
    }
    return { kind: TypeKind.NETWORK, hasConst: false, constValue: undefined };
  }

  if (calleeName === 'train') {
    const arg0 = argTypes[0];
    const arg2 = argTypes[2];
    if (!arg0) {
      err(errors, lineHint, 'train(spec, data, opts) requires an architecture argument.');
    } else if (arg0.hasConst) {
      validateArchitectureSpec(arg0.constValue, lineHint, errors);
    }
    if (arg2 && arg2.hasConst) validateTrainingOptions(arg2.constValue, lineHint, errors);
    return { kind: TypeKind.NETWORK, hasConst: false, constValue: undefined };
  }

  if (calleeName === 'tensor') return { kind: TypeKind.TENSOR, hasConst: false, constValue: undefined };
  if (calleeName === 'forward' || calleeName === 'predict') return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
  if (calleeName === 'len') return { kind: TypeKind.NUMBER, hasConst: false, constValue: undefined };
  if (calleeName === 'range') return { kind: TypeKind.LIST, hasConst: false, constValue: undefined };
  if (calleeName === 'await') return argTypes[0] || { kind: TypeKind.ANY, hasConst: false, constValue: undefined };

  return { kind: TypeKind.ANY, hasConst: false, constValue: undefined };
}

function walkStmt(node, scope, errors) {
  const line = node.line || 0;
  switch (node.type) {
    case 'Let': {
      const t = inferType(node.value, scope, errors, line);
      scope.define(node.name, t);
      return;
    }
    case 'Set': {
      const t = inferType(node.value, scope, errors, line);
      if (node.chain && node.chain.length > 0) {
        // Nested mutation invalidates constness on the root variable.
        scope.assign(node.name, { kind: TypeKind.OBJECT, hasConst: false, constValue: undefined });
      } else {
        scope.assign(node.name, t);
      }
      return;
    }
    case 'If': {
      inferType(node.cond, scope, errors, line);
      const leftScope = new Scope(scope);
      for (const s of node.consequent) walkStmt(s, leftScope, errors);
      if (node.alternate) {
        const rightScope = new Scope(scope);
        for (const s of node.alternate) walkStmt(s, rightScope, errors);
      }
      return;
    }
    case 'While': {
      inferType(node.cond, scope, errors, line);
      const bodyScope = new Scope(scope);
      for (const s of node.body) walkStmt(s, bodyScope, errors);
      return;
    }
    case 'For': {
      inferType(node.from, scope, errors, line);
      inferType(node.to, scope, errors, line);
      if (node.by) inferType(node.by, scope, errors, line);
      const bodyScope = new Scope(scope);
      bodyScope.define(node.name, { kind: TypeKind.NUMBER, hasConst: false, constValue: undefined });
      for (const s of node.body) walkStmt(s, bodyScope, errors);
      return;
    }
    case 'FnDecl': {
      scope.define(node.name, { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
      const fnScope = new Scope(scope);
      for (const p of node.params) fnScope.define(p, { kind: TypeKind.ANY, hasConst: false, constValue: undefined });
      for (const s of node.body) walkStmt(s, fnScope, errors);
      return;
    }
    case 'Return':
      if (node.value) inferType(node.value, scope, errors, line);
      return;
    case 'Print':
      inferType(node.value, scope, errors, line);
      return;
    case 'ExpressionStatement':
      inferType(node.expression, scope, errors, line);
      return;
    default:
      return;
  }
}

function typeCheck(ast) {
  const errors = [];
  if (!ast || ast.type !== 'Program') return { ok: false, errors: [{ line: 0, message: 'Invalid AST root.' }] };

  const root = new Scope(null);
  // Builtins exposed by interpreter stdlib.
  root.define('build', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('train', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('predict', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('forward', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('tensor', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('await', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('len', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });
  root.define('range', { kind: TypeKind.FUNCTION, hasConst: false, constValue: undefined });

  for (const stmt of ast.body) walkStmt(stmt, root, errors);

  return { ok: errors.length === 0, errors };
}

module.exports = { typeCheck };
