'use strict';

const http = require('http');
const os = require('os');
const { EventEmitter } = require('events');
const { infer } = require('../engine/trainer');

function getHostIp() {
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name]) {
      if (info.family === 'IPv4' && !info.internal) return info.address;
    }
  }
  return '127.0.0.1';
}

class ApiServer extends EventEmitter {
  constructor(storage) {
    super();
    this.storage = storage;
    this.servers = new Map(); // id -> { server, port, startedAt }
  }

  listAll() {
    const out = [];
    for (const [id, rec] of this.servers) {
      out.push({ id, port: rec.port, startedAt: rec.startedAt, url: `http://${getHostIp()}:${rec.port}` });
    }
    return out;
  }

  status(id) {
    const rec = this.servers.get(id);
    if (!rec) return { running: false };
    return { running: true, port: rec.port, url: `http://${getHostIp()}:${rec.port}` };
  }

  start(id, port) {
    if (this.servers.has(id)) {
      const rec = this.servers.get(id);
      return { running: true, port: rec.port, url: `http://${getHostIp()}:${rec.port}` };
    }
    const net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (!net.state || net.stateLocked) throw new Error('Network has no trained state');
    const useePort = Number(port) || 0;

    const server = http.createServer((req, res) => {
      const url = req.url || '/';
      if (req.method === 'GET' && (url === '/' || url === '/info')) {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({
          name: net.name,
          id: net.id,
          kind: net.architecture.kind,
          description: net.description,
          inputSpec: this._inputSpec(net)
        }));
        this.emit('log', { id, line: `GET ${url}` });
        return;
      }
      if (req.method === 'POST' && url === '/predict') {
        let body = '';
        req.on('data', c => body += c);
        req.on('end', () => {
          try {
            const payload = body ? JSON.parse(body) : {};
            const freshNet = this.storage.getNetwork(id);
            if (!freshNet || !freshNet.state) throw new Error('Model no longer available');
            const result = infer(freshNet, payload);
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(result));
            this.emit('log', { id, line: `POST /predict ok` });
          } catch (e) {
            res.statusCode = 400;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ error: e.message }));
            this.emit('log', { id, line: `POST /predict ERROR: ${e.message}` });
          }
        });
        return;
      }
      res.statusCode = 404;
      res.end('Not Found');
    });

    server.on('error', (e) => {
      this.emit('log', { id, line: `server error: ${e.message}` });
      this.servers.delete(id);
    });

    return new Promise((resolve, reject) => {
      server.listen(useePort, () => {
        const actualPort = server.address().port;
        this.servers.set(id, { server, port: actualPort, startedAt: Date.now() });
        this.emit('log', { id, line: `listening on ${actualPort}` });
        resolve({ running: true, port: actualPort, url: `http://${getHostIp()}:${actualPort}` });
      });
      server.once('error', reject);
    });
  }

  stop(id) {
    const rec = this.servers.get(id);
    if (!rec) return { running: false };
    rec.server.close();
    this.servers.delete(id);
    this.emit('log', { id, line: 'stopped' });
    return { running: false };
  }

  stopAll() {
    for (const [id, rec] of this.servers) {
      try { rec.server.close(); } catch (e) {}
    }
    this.servers.clear();
  }

  _inputSpec(net) {
    const a = net.architecture;
    if (a.kind === 'classifier' || a.kind === 'mlp') return { type: 'vector', length: a.inputDim };
    if (a.kind === 'regressor') return { type: 'vector', length: a.inputDim, outputLength: a.outputDim };
    if (a.kind === 'charLM') return { type: 'text', fields: ['prompt', 'maxTokens', 'temperature', 'topK'] };
    return { type: 'unknown' };
  }
}

module.exports = { ApiServer };
