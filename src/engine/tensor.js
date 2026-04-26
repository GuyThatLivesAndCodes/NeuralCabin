'use strict';

const fallback = require('./tensor-js');
const { loadNativeTensorBackend } = require('./tensor-native-loader');

const backend = loadNativeTensorBackend(fallback);

module.exports = backend.api;
module.exports.__backend = backend.meta;
