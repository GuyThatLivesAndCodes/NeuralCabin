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

// Conversation sessions are kept entirely in process memory. Restarting the
// API server (or the app) clears them. We cap each session's history so a
// long-running client can't grow it unbounded.
const MAX_SESSION_TURNS = 64;
const MAX_SESSIONS_PER_MODEL = 256;
const SESSION_TTL_MS = 60 * 60 * 1000; // 1 hour idle

class ApiServer extends EventEmitter {
  constructor(storage) {
    super();
    this.storage = storage;
    this.servers = new Map(); // id -> { server, port, startedAt, sessions }
  }

  _getOrCreateSession(rec, sessionId) {
    if (!sessionId) sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    let s = rec.sessions.get(sessionId);
    if (!s) {
      // Evict oldest if at cap.
      if (rec.sessions.size >= MAX_SESSIONS_PER_MODEL) {
        let oldestKey = null, oldestAt = Infinity;
        for (const [k, v] of rec.sessions) if (v.lastSeen < oldestAt) { oldestAt = v.lastSeen; oldestKey = k; }
        if (oldestKey) rec.sessions.delete(oldestKey);
      }
      s = { history: [], system: '', createdAt: Date.now(), lastSeen: Date.now() };
      rec.sessions.set(sessionId, s);
    }
    s.lastSeen = Date.now();
    return { sessionId, session: s };
  }

  _gcSessions(rec) {
    const now = Date.now();
    for (const [k, v] of rec.sessions) if (now - v.lastSeen > SESSION_TTL_MS) rec.sessions.delete(k);
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

    const sessions = new Map(); // sessionId -> { history, system, createdAt, lastSeen }
    const recRefHolder = { sessions };

    const readBody = (req) => new Promise((resolve, reject) => {
      let body = '';
      req.on('data', c => body += c);
      req.on('end', () => resolve(body));
      req.on('error', reject);
    });

    const server = http.createServer(async (req, res) => {
      const url = req.url || '/';
      res.setHeader('Content-Type', 'application/json');
      if (req.method === 'GET' && (url === '/' || url === '/info')) {
        res.end(JSON.stringify({
          name: net.name,
          id: net.id,
          kind: net.architecture.kind,
          description: net.description,
          inputSpec: this._inputSpec(net),
          // Surface multi-turn capability so clients know whether /chat is meaningful.
          chat: !!net.architecture.isChat
        }));
        this.emit('log', { id, line: `GET ${url}` });
        return;
      }
      if (req.method === 'POST' && url === '/predict') {
        try {
          const body = await readBody(req);
          const payload = body ? JSON.parse(body) : {};
          const freshNet = this.storage.getNetwork(id);
          if (!freshNet || !freshNet.state) throw new Error('Model no longer available');
          const result = infer(freshNet, payload);
          res.end(JSON.stringify(result));
          this.emit('log', { id, line: `POST /predict ok` });
        } catch (e) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: e.message }));
          this.emit('log', { id, line: `POST /predict ERROR: ${e.message}` });
        }
        return;
      }
      // Stateful chat: client passes { message, sessionId?, system? } and we
      // keep the running history in memory keyed by sessionId. New sessionIds
      // are minted on the fly. This is for naive clients that don't want to
      // manage history themselves — rich clients should use /predict with an
      // explicit `history` array.
      if (req.method === 'POST' && url === '/chat') {
        try {
          const body = await readBody(req);
          const payload = body ? JSON.parse(body) : {};
          const freshNet = this.storage.getNetwork(id);
          if (!freshNet || !freshNet.state) throw new Error('Model no longer available');
          if (!freshNet.architecture.isChat) throw new Error('Model is not a chat model — train it on chat samples or use /predict');
          this._gcSessions(recRefHolder);
          const { sessionId, session } = this._getOrCreateSession(recRefHolder, payload.sessionId);
          const message = String(payload.message ?? payload.prompt ?? '');
          if (!message) throw new Error('"message" is required');
          if (typeof payload.system === 'string') session.system = payload.system;
          const result = infer(freshNet, {
            history: session.history,
            prompt: message,
            system: session.system,
            maxTokens: payload.maxTokens,
            temperature: payload.temperature,
            topK: payload.topK
          });
          // Append both the user turn and the model's reply so subsequent
          // calls in this session see the full thread.
          session.history.push({ role: 'user', content: message });
          session.history.push({ role: 'assistant', content: result.text });
          // Cap history length (drop oldest pairs).
          while (session.history.length > MAX_SESSION_TURNS) session.history.shift();
          res.end(JSON.stringify({
            sessionId,
            reply: result.text,
            history: session.history.slice()
          }));
          this.emit('log', { id, line: `POST /chat ok (session=${sessionId.slice(0, 14)}…, turns=${session.history.length})` });
        } catch (e) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: e.message }));
          this.emit('log', { id, line: `POST /chat ERROR: ${e.message}` });
        }
        return;
      }
      // Reset a chat session. Useful when a client wants to start over without
      // minting a brand new sessionId.
      if (req.method === 'POST' && url === '/chat/reset') {
        try {
          const body = await readBody(req);
          const payload = body ? JSON.parse(body) : {};
          if (payload.sessionId && recRefHolder.sessions.has(payload.sessionId)) {
            recRefHolder.sessions.delete(payload.sessionId);
            res.end(JSON.stringify({ ok: true, cleared: payload.sessionId }));
          } else {
            res.end(JSON.stringify({ ok: true, cleared: null }));
          }
          this.emit('log', { id, line: `POST /chat/reset` });
        } catch (e) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: e.message }));
        }
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
        this.servers.set(id, { server, port: actualPort, startedAt: Date.now(), sessions });
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
    if (a.kind === 'charLM') {
      const base = { type: 'text', fields: ['prompt', 'maxTokens', 'temperature', 'topK'] };
      if (a.isChat) {
        base.chatFields = ['history', 'messages', 'system'];
        base.chatEndpoints = { stateful: '/chat', resetSession: '/chat/reset' };
      }
      return base;
    }
    return { type: 'unknown' };
  }
}

module.exports = { ApiServer };
