'use strict';

const path = require('path');

const REQUIRED_EXPORTS = [
  'Tensor',
  'tensor', 'zeros', 'ones', 'randn', 'rngFromSeed',
  'add', 'sub', 'mul', 'matmul',
  'relu', 'leakyRelu', 'tanh', 'sigmoid', 'gelu', 'softmax',
  'softmaxCrossEntropy', 'mseLoss',
  'dropout', 'embedding', 'sumAll', 'mean',
  'shapeSize', 'sameShape'
];

function hasCompatibleApi(mod) {
  if (!mod || typeof mod !== 'object') return false;
  for (const key of REQUIRED_EXPORTS) {
    if (!(key in mod)) return false;
  }
  return true;
}

function normalizeMode(mode) {
  return String(mode || '').trim().toLowerCase();
}

function loadNativeTensorBackend(fallback) {
  const mode = normalizeMode(process.env.NEURALCABIN_ENGINE_BACKEND || 'auto');
  if (mode === 'js') {
    return { api: fallback, meta: { mode: 'js', reason: 'forced-js' } };
  }

  const candidates = [];
  if (process.env.NEURALCABIN_NATIVE_BINDING) {
    candidates.push(process.env.NEURALCABIN_NATIVE_BINDING);
  }
  candidates.push(
    path.resolve(process.cwd(), 'native', 'rust-engine', 'artifacts', 'neuralcabin_native.node'),
    path.resolve(process.cwd(), 'native', 'rust-engine', 'neuralcabin-node', 'index.js'),
    '@neuralcabin/engine-native'
  );

  for (const candidate of candidates) {
    try {
      const loaded = require(candidate);
      const api = loaded && loaded.api ? loaded.api : loaded;
      if (!hasCompatibleApi(api)) continue;
      return {
        api,
        meta: {
          mode: 'rust',
          reason: 'loaded',
          module: candidate,
          version: loaded.version || null
        }
      };
    } catch (err) {
      if (mode === 'rust') {
        throw new Error(`Rust tensor backend requested but failed to load "${candidate}": ${err.message}`);
      }
    }
  }

  if (mode === 'rust') {
    throw new Error('Rust tensor backend requested but no compatible native binding was found.');
  }

  return { api: fallback, meta: { mode: 'js', reason: 'fallback' } };
}

module.exports = { loadNativeTensorBackend };
