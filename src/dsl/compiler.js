'use strict';

const { tokenize } = require('./lexer');
const { parse } = require('./parser');
const { typeCheck } = require('./typecheck');

function formatDiagnostics(diags) {
  return diags
    .map(d => `line ${d.line || 0}: ${d.message}`)
    .join('\n');
}

function compileNeuralScript(code) {
  const tokens = tokenize(code);
  const ast = parse(tokens);
  const diagnostics = typeCheck(ast);
  if (!diagnostics.ok) {
    const e = new Error(`NeuralScript type check failed:\n${formatDiagnostics(diagnostics.errors)}`);
    e.diagnostics = diagnostics.errors;
    throw e;
  }
  return { tokens, ast, diagnostics };
}

module.exports = { compileNeuralScript };
