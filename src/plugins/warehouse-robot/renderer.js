// Warehouse Robot plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  const GRID = 8;
  const CELL = 54;
  const CW   = GRID * CELL;  // 432

  // ── Palette ───────────────────────────────────────────────────────────────
  const COL = {
    bg:        '#111111',
    gridLine:  '#222222',
    target:    '#1e4d1e',
    targetRing:'#3a8a3a',
    box:       '#7c5522',
    boxStroke: '#5a3a10',
    boxOnTgt:  '#388e3c',
    boxOnStr:  '#1b5e20',
    robot:     '#1976d2',
    robotHi:   '#90caf9',
    robotInfer:'#e91e63',
    robotInfHi:'#f8bbd0',
  };

  // ── Shared drawing helper ─────────────────────────────────────────────────

  function drawGridState(ctx, s, robotColor, robotHiColor) {
    ctx.clearRect(0, 0, CW, CW);

    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        ctx.fillStyle = COL.bg;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
        ctx.strokeStyle = COL.gridLine;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * CELL, r * CELL, CELL, CELL);
      }
    }

    if (!s) return;

    for (const [tr, tc] of s.targets) {
      const cx = tc * CELL + CELL / 2, cy = tr * CELL + CELL / 2;
      ctx.fillStyle = COL.target;
      ctx.fillRect(tc * CELL, tr * CELL, CELL, CELL);
      ctx.strokeStyle = COL.targetRing;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, CELL * 0.32, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
      ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
      ctx.stroke();
    }

    for (const [br, bc] of s.boxes) {
      const onTgt = s.targets.some(t => t[0] === br && t[1] === bc);
      const x = bc * CELL, y = br * CELL, pad = 7;
      ctx.fillStyle   = onTgt ? COL.boxOnTgt  : COL.box;
      ctx.strokeStyle = onTgt ? COL.boxOnStr  : COL.boxStroke;
      ctx.lineWidth   = 1.5;
      ctx.fillRect  (x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
      ctx.strokeRect(x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
      const mx = x + CELL / 2, my = y + CELL / 2;
      ctx.strokeStyle = onTgt ? '#a5d6a7' : '#ffcc80';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5);
      ctx.moveTo(mx + 5, my - 5); ctx.lineTo(mx - 5, my + 5);
      ctx.stroke();
    }

    const [rr, rc] = s.robot;
    const rx = rc * CELL + CELL / 2, ry = rr * CELL + CELL / 2;
    ctx.fillStyle = robotColor;
    ctx.beginPath();
    ctx.arc(rx, ry, CELL * 0.27, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = robotHiColor;
    ctx.beginPath();
    ctx.arc(rx, ry, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawRewardChart(svgEl, hist) {
    if (!svgEl || !hist || hist.length < 2) { if (svgEl) svgEl.innerHTML = ''; return; }
    const W = 260, H = 70, pad = 4;
    const min = Math.min(...hist), max = Math.max(...hist);
    const range = max === min ? 1 : max - min;
    const pts = hist.map((v, i) => {
      const x = pad + (i / (hist.length - 1)) * (W - 2 * pad);
      const y = H - pad - ((v - min) / range) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const zy = Math.max(pad, Math.min(H - pad,
      H - pad - ((0 - min) / range) * (H - 2 * pad))).toFixed(1);
    svgEl.innerHTML =
      `<line x1="${pad}" y1="${zy}" x2="${W - pad}" y2="${zy}" stroke="#222" stroke-width="1"/>` +
      `<polyline points="${pts}" fill="none" stroke="#4caf50" stroke-width="1.5" stroke-linejoin="round"/>`;
  }

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'warehouse-robot',
    name: 'Warehouse Robot (Q-Learning)',
    kind: 'classifier',
    pluginKind: 'warehouse-robot',
    desc: 'A DQN agent learns to push 3 boxes onto their target squares through trial and error. No training data required — launch the simulation from the Train tab.',
    arch: {
      kind: 'classifier', pluginKind: 'warehouse-robot',
      inputDim: 14, outputDim: 4,
      hidden: [128, 64], activation: 'relu', dropout: 0,
    },
    training: { optimizer: 'adam', learningRate: 0.001, batchSize: 64, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train settings — relabel standard training fields ─────────────────────
  api.registerTrainSettings('warehouse-robot', {
    lr:        { label: 'Learning rate',   hint: 'DQN optimizer learning rate (default 0.001)' },
    bs:        { label: 'Replay batch',    hint: 'Experiences sampled per training step (default 64)' },
    epochs:    { label: 'Max episodes',    hint: 'Training stops after this many episodes (0 = unlimited)' },
    seed:      { label: 'Random seed',     hint: 'Seed for initial environment layout' },
    workers:   { hidden: true },
    optimizer: { hidden: true },
    sectionHint: 'DQN agent settings — applied when the simulation starts.',
  });

  // ── Train editor (training data section) ──────────────────────────────────
  api.registerTrainEditor('warehouse-robot', function (root) {
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#0d2b0d;border:1px solid #2d5a2d;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#4caf50;margin-bottom:5px;">Q-Learning — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            A Deep Q-Network generates its own experience by interacting with the 8×8 grid.
            The agent starts exploring randomly (ε = 1.0) and shifts to a learned policy as
            epsilon decays to 0.05 over ~13 000 steps.
            Configure learning rate and replay batch in the <strong style="color:#ccc;">Training settings</strong> above.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;line-height:1.6;">
          Architecture: <strong style="color:#aaa;">14 → [128, 64] → 4</strong><br>
          State: robot pos + 3 box positions + 3 target positions (normalized)<br>
          Actions: UP / DOWN / LEFT / RIGHT<br><br>
          Use the <strong style="color:#ccc;">Train</strong> tab to run the live simulation.<br>
          The <strong style="color:#ccc;">Infer</strong> tab shows the greedy policy with a noise slider.
        </div>
      </div>
    `;
  });

  // ── Train renderer — full DQN training simulation ─────────────────────────
  api.registerTrainRenderer('warehouse-robot', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _initialized = false;

    const t       = (network && network.training) || {};
    const cfgLR   = t.learningRate || 0.001;
    const cfgBS   = (t.batchSize | 0) || 64;
    const cfgSeed = (t.seed || 42) >>> 0;

    root.innerHTML = `
      <div class="panel" style="max-width:860px;">
        <h2>Warehouse Robot — Q-Learning</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          DQN agent explores an 8×8 grid and learns to push all 3 boxes onto target rings.
          LR: ${cfgLR} · Batch: ${cfgBS} · Epsilon decays 1.0 → 0.05 over ~13 000 steps.
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="wh-canvas" width="${CW}" height="${CW}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#111;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;">
              <button class="btn primary" id="wh-start">Start</button>
              <button class="btn"         id="wh-pause">Pause</button>
              <button class="btn"         id="wh-reset">Reset agent</button>
            </div>
            <div style="margin-top:8px;display:flex;align-items:center;gap:8px;font-size:12px;color:#666;">
              Steps / frame:
              <input id="wh-speed" type="range" min="1" max="50" value="10" style="width:80px;">
              <span id="wh-speed-val">10</span>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Episode</div><div class="v" id="wh-ep">0</div></div>
              <div class="kpi"><div class="k">Epsilon ε</div><div class="v" id="wh-eps">1.0000</div></div>
              <div class="kpi"><div class="k">Ep reward</div><div class="v" id="wh-rew">0.00</div></div>
              <div class="kpi"><div class="k">Best reward</div><div class="v" id="wh-best">—</div></div>
            </div>

            <div style="background:#0d2b0d;border:1px solid #2d4a2d;border-radius:6px;padding:10px 14px;text-align:center;">
              <div style="font-size:11px;color:#777;margin-bottom:4px;">Boxes on target</div>
              <div id="wh-ontgt" style="font-size:28px;font-weight:700;color:#4caf50;">0 / 3</div>
            </div>

            <div class="section">
              <h3>Episode reward history</h3>
              <svg id="wh-chart" viewBox="0 0 260 70" preserveAspectRatio="none"
                style="width:100%;height:70px;display:block;background:#0d0d0d;border-radius:4px;"></svg>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.7;">
              <strong style="color:#666;">Legend</strong><br>
              <span style="color:#7c5522;">■</span> Box &nbsp;&nbsp;
              <span style="color:#388e3c;">■</span> Box on target &nbsp;&nbsp;
              <span style="color:#1976d2;">●</span> Robot<br>
              <span style="color:#3a8a3a;">◎</span> Target ring<br><br>
              Rewards: +10 per box placed · +50 all done<br>
              −0.5 wall · −0.4 blocked · −0.01/step
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('wh-canvas');
    const ctx    = canvas.getContext('2d');

    function updateStats(s) {
      document.getElementById('wh-ep').textContent   = s.episode;
      document.getElementById('wh-eps').textContent  = s.epsilon.toFixed(4);
      document.getElementById('wh-rew').textContent  = s.epReward.toFixed(2);
      document.getElementById('wh-best').textContent = s.bestReward == null ? '—' : s.bestReward.toFixed(2);
      document.getElementById('wh-ontgt').textContent = `${s.onTarget} / 3`;
      drawRewardChart(document.getElementById('wh-chart'), s.rewardHistory);
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const n = parseInt(document.getElementById('wh-speed').value) || 10;
      try {
        const s = await nb.invoke('warehouse-robot:step', n);
        if (s) { drawGridState(ctx, s, COL.robot, COL.robotHi); updateStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await nb.invoke('warehouse-robot:init', { lr: cfgLR, batchSize: cfgBS, seed: cfgSeed });
        if (r && r.error) { console.error('[warehouse-robot]', r.error); return; }
        _initialized = true;
      } else {
        await nb.invoke('warehouse-robot:start');
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseSim() {
      _running = false;
      nb.invoke('warehouse-robot:stop');
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function resetSim() {
      pauseSim();
      await nb.invoke('warehouse-robot:reset');
      _initialized = true;
      drawGridState(ctx, null, COL.robot, COL.robotHi);
      ['wh-ep', 'wh-eps', 'wh-rew', 'wh-best'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        if (id === 'wh-eps') el.textContent = '1.0000';
        else if (id === 'wh-rew') el.textContent = '0.00';
        else if (id === 'wh-ep') el.textContent = '0';
        else el.textContent = '—';
      });
      document.getElementById('wh-ontgt').textContent = '0 / 3';
      const svg = document.getElementById('wh-chart');
      if (svg) svg.innerHTML = '';
    }

    document.getElementById('wh-start').addEventListener('click', startSim);
    document.getElementById('wh-pause').addEventListener('click', pauseSim);
    document.getElementById('wh-reset').addEventListener('click', resetSim);

    const slider = document.getElementById('wh-speed');
    slider.addEventListener('input', () => {
      document.getElementById('wh-speed-val').textContent = slider.value;
    });

    nb.invoke('warehouse-robot:getState').then(s => {
      if (s) { _initialized = true; drawGridState(ctx, s, COL.robot, COL.robotHi); updateStats(s); }
      else    { drawGridState(ctx, null, COL.robot, COL.robotHi); }
    }).catch(() => drawGridState(ctx, null, COL.robot, COL.robotHi));
  });

  // ── Inference renderer — greedy policy + noise slider ────────────────────
  api.registerInferenceRenderer('warehouse-robot', function (root, network, nb) {
    let _raf      = null;
    let _running  = false;
    let _ready    = false;
    let _noiseStd = 0;

    root.innerHTML = `
      <div class="panel" style="max-width:860px;">
        <h2>Warehouse Robot — Greedy Policy</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          The trained DQN runs greedy (ε = 0). Add state noise to see how robust the policy is.
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="wh-i-canvas" width="${CW}" height="${CW}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#111;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;">
              <button class="btn primary" id="wh-i-start">Start</button>
              <button class="btn"         id="wh-i-pause">Pause</button>
              <button class="btn"         id="wh-i-rerun">New layout</button>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Trained eps</div><div class="v" id="wh-i-ep">—</div></div>
              <div class="kpi"><div class="k">Best reward</div><div class="v" id="wh-i-best">—</div></div>
              <div class="kpi"><div class="k">Episodes done</div><div class="v" id="wh-i-done">0</div></div>
              <div class="kpi"><div class="k">Ep reward</div><div class="v" id="wh-i-rew">0.00</div></div>
            </div>

            <div style="background:#0d2b0d;border:1px solid #2d4a2d;border-radius:6px;padding:10px 14px;text-align:center;">
              <div style="font-size:11px;color:#777;margin-bottom:4px;">Boxes on target</div>
              <div id="wh-i-ontgt" style="font-size:28px;font-weight:700;color:#4caf50;">0 / 3</div>
            </div>

            <div class="section">
              <h3>State noise</h3>
              <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#aaa;">
                <span style="min-width:18px;">0</span>
                <input id="wh-i-noise" type="range" min="0" max="30" value="0" style="flex:1;">
                <span style="min-width:28px;">0.30</span>
                <span id="wh-i-noise-val" style="min-width:36px;text-align:right;color:#4caf50;">0.00</span>
              </div>
              <div style="font-size:11px;color:#555;margin-top:4px;">
                Gaussian noise std added to all 14 state dimensions. Drag right to stress-test.
              </div>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.7;">
              <span style="color:#e91e63;">●</span> Robot (greedy) &nbsp;&nbsp;
              <span style="color:#388e3c;">■</span> Box on target<br><br>
              The agent resets to a new random layout after each episode.<br>
              Switch to the <strong style="color:#666;">Train</strong> tab to keep training.
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('wh-i-canvas');
    const ctx    = canvas.getContext('2d');

    function showPlaceholder(msg) {
      drawGridState(ctx, null, COL.robotInfer, COL.robotInfHi);
      ctx.fillStyle    = '#444';
      ctx.font         = '13px monospace';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(msg, CW / 2, CW / 2);
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'alphabetic';
    }

    function updateInferStats(s) {
      if (!s) return;
      document.getElementById('wh-i-ontgt').textContent = `${s.onTarget} / 3`;
      document.getElementById('wh-i-rew').textContent   = s.epReward.toFixed(2);
      document.getElementById('wh-i-done').textContent  = s.episodesDone;
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      try {
        const s = await nb.invoke('warehouse-robot:inferStep', _noiseStd);
        if (s) { drawGridState(ctx, s, COL.robotInfer, COL.robotInfHi); updateInferStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startInfer() {
      if (!_ready) {
        const r = await nb.invoke('warehouse-robot:inferInit');
        if (!r || r.error) {
          showPlaceholder(r ? r.error : 'No trained agent — run the Train tab first.');
          return;
        }
        document.getElementById('wh-i-ep').textContent   = r.episode || '—';
        document.getElementById('wh-i-best').textContent = r.bestReward != null ? r.bestReward.toFixed(2) : '—';
        _ready = true;
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseInfer() {
      _running = false;
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function rerun() {
      const wasRunning = _running;
      pauseInfer();
      _ready = false;
      document.getElementById('wh-i-rew').textContent  = '0.00';
      document.getElementById('wh-i-ontgt').textContent = '0 / 3';
      const r = await nb.invoke('warehouse-robot:inferInit');
      if (!r || r.error) {
        showPlaceholder(r ? r.error : 'No trained agent — run the Train tab first.');
        return;
      }
      document.getElementById('wh-i-ep').textContent   = r.episode || '—';
      document.getElementById('wh-i-best').textContent = r.bestReward != null ? r.bestReward.toFixed(2) : '—';
      _ready = true;
      if (wasRunning) {
        _running = true;
        _raf = requestAnimationFrame(tick);
      }
    }

    document.getElementById('wh-i-start').addEventListener('click', startInfer);
    document.getElementById('wh-i-pause').addEventListener('click', pauseInfer);
    document.getElementById('wh-i-rerun').addEventListener('click', rerun);

    const noiseSlider = document.getElementById('wh-i-noise');
    noiseSlider.addEventListener('input', () => {
      _noiseStd = parseInt(noiseSlider.value) / 100;
      document.getElementById('wh-i-noise-val').textContent = _noiseStd.toFixed(2);
    });

    showPlaceholder('Press Start to view the greedy policy.');
  });

})(api);
