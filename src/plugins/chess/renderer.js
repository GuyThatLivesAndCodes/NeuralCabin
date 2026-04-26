// Chess plugin renderer — evaluated in the renderer process via new Function('api', code)(api)
(function (api) {
  'use strict';

  const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

  // Filled Unicode piece symbols — same shape for both colors, CSS controls fill/stroke
  const PIECE_SYMBOL = { 1:'♟', 2:'♞', 3:'♝', 4:'♜', 5:'♛', 6:'♚' };

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'chess',
    name: 'Chess Move Predictor',
    kind: 'classifier',
    pluginKind: 'chess',
    desc: 'Train a neural network to predict chess moves. Upload UCI game files, then play against the AI for both sides.',
    arch: {
      kind: 'classifier',
      pluginKind: 'chess',
      inputDim: 73,
      outputDim: 4096,
      hidden: [256, 128, 64],
      activation: 'relu',
      dropout: 0.1
    },
    training: { optimizer: 'adam', learningRate: 0.001, batchSize: 64, epochs: 30, seed: 42 },
    trainingData: { samples: [] }
  });

  // ── FEN parser (renderer-side, for board rendering only) ──────────────────
  const PIECE_VAL = { P:1, N:2, B:3, R:4, Q:5, K:6, p:-1, n:-2, b:-3, r:-4, q:-5, k:-6 };
  function parseFENBoard(fen) {
    const board = new Array(64).fill(0);
    let rank = 7, file = 0;
    for (const c of fen.split(' ')[0]) {
      if (c === '/') { rank--; file = 0; }
      else if (c >= '1' && c <= '8') { file += parseInt(c); }
      else { board[rank * 8 + file] = PIECE_VAL[c] || 0; file++; }
    }
    const turn = (fen.split(' ')[1] || 'w') === 'w' ? 'white' : 'black';
    return { board, turn };
  }

  function sqName(idx) { return 'abcdefgh'[idx % 8] + (Math.floor(idx / 8) + 1); }
  function nameToSq(s) { return (parseInt(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97); }

  // Board palette
  const SQ_LIGHT     = '#d9d9d9'; // light gray
  const SQ_DARK      = '#4a7c40'; // grass green
  const SQ_HL_FROM   = '#f5f534'; // bright yellow — last-move from
  const SQ_HL_TO     = '#d4c829'; // slightly darker yellow — last-move to
  const SQ_HL_SEL    = '#f5f534'; // bright yellow — selected piece
  const SQ_HL_DEST   = '#6ab84a'; // lighter green — legal-move dot

  // ── Promotion dialog ─────────────────────────────────────────────────────
  // Shows a picker overlay on the board; resolves with the chosen piece char.
  function showPromotionDialog(boardWrap, isWhite) {
    return new Promise(resolve => {
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.75);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:100;border-radius:3px;gap:12px;';

      const title = document.createElement('div');
      title.style.cssText = 'color:#fff;font-size:13px;font-weight:700;letter-spacing:0.03em;';
      title.textContent = 'Promote pawn to:';
      overlay.appendChild(title);

      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:10px;';

      const promos = [
        { char: 'q', sym: '♛', label: 'Queen'  },
        { char: 'r', sym: '♜', label: 'Rook'   },
        { char: 'b', sym: '♝', label: 'Bishop' },
        { char: 'n', sym: '♞', label: 'Knight' }
      ];

      for (const p of promos) {
        const btn = document.createElement('div');
        btn.style.cssText = `width:66px;height:66px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;background:#1e1e1e;border:2px solid #444;border-radius:8px;cursor:pointer;transition:border-color 0.1s;`;
        const sym = document.createElement('span');
        sym.textContent = p.sym;
        sym.style.cssText = `font-size:32px;line-height:1;color:${isWhite ? '#ffffff' : '#888888'};-webkit-text-stroke:1.5px #111111;text-shadow:0 1px 3px rgba(0,0,0,0.6);`;
        const lbl = document.createElement('span');
        lbl.textContent = p.label;
        lbl.style.cssText = 'font-size:10px;color:#aaa;';
        btn.appendChild(sym);
        btn.appendChild(lbl);
        btn.addEventListener('mouseenter', () => { btn.style.borderColor = '#7ec8e3'; });
        btn.addEventListener('mouseleave', () => { btn.style.borderColor = '#444'; });
        btn.addEventListener('click', () => { overlay.remove(); resolve(p.char); });
        row.appendChild(btn);
      }

      overlay.appendChild(row);

      // Board wrap needs position:relative so the absolute overlay is anchored to it
      const prevPos = boardWrap.style.position;
      if (!prevPos || prevPos === 'static') boardWrap.style.position = 'relative';
      boardWrap.appendChild(overlay);
    });
  }

  // ── Board renderer ────────────────────────────────────────────────────────
  function renderBoard(container, fen, highlights, onSquareClick) {
    const { board, turn } = parseFENBoard(fen);
    container.innerHTML = '';
    container.style.cssText = 'display:grid;grid-template-columns:repeat(8,46px);grid-template-rows:repeat(8,46px);border:2px solid #333;border-radius:3px;overflow:hidden;width:368px;height:368px;box-shadow:0 4px 16px rgba(0,0,0,0.5);';

    for (let displayRank = 7; displayRank >= 0; displayRank--) {
      for (let file = 0; file < 8; file++) {
        const idx = displayRank * 8 + file;
        const light = (displayRank + file) % 2 === 1;
        const cell = document.createElement('div');
        const hl = highlights && highlights[idx];

        let bg = light ? SQ_LIGHT : SQ_DARK;
        if (hl === 'from') bg = SQ_HL_FROM;
        if (hl === 'to')   bg = SQ_HL_TO;
        if (hl === 'sel')  bg = SQ_HL_SEL;

        cell.style.cssText = `width:46px;height:46px;display:flex;align-items:center;justify-content:center;font-size:34px;line-height:1;cursor:pointer;background:${bg};user-select:none;position:relative;`;

        const p = board[idx];
        if (p !== 0) {
          cell.textContent = PIECE_SYMBOL[Math.abs(p)] || '';
          // White pieces: white fill + black outline; Black pieces: medium-gray fill + black outline
          cell.style.color = p > 0 ? '#ffffff' : '#888888';
          cell.style.webkitTextStroke = '1.5px #111111';
          cell.style.textShadow = '0 1px 3px rgba(0,0,0,0.6)';
          // Capture-available ring on occupied legal-move destinations
          if (hl === 'dest') {
            cell.style.boxShadow = `inset 0 0 0 4px ${SQ_HL_DEST}cc`;
          }
        } else if (hl === 'dest') {
          // Legal-move dot on empty squares
          const dot = document.createElement('div');
          dot.style.cssText = `width:14px;height:14px;border-radius:50%;background:${SQ_HL_DEST};opacity:0.8;pointer-events:none;`;
          cell.appendChild(dot);
        }

        // Rank/file coordinate labels
        const labelColor = light ? 'rgba(80,80,80,0.7)' : 'rgba(210,230,210,0.7)';
        if (file === 0) {
          const lbl = document.createElement('span');
          lbl.style.cssText = `position:absolute;top:2px;left:3px;font-size:9px;color:${labelColor};font-weight:bold;pointer-events:none;`;
          lbl.textContent = displayRank + 1;
          cell.appendChild(lbl);
        }
        if (displayRank === 0) {
          const lbl = document.createElement('span');
          lbl.style.cssText = `position:absolute;bottom:2px;right:3px;font-size:9px;color:${labelColor};font-weight:bold;pointer-events:none;`;
          lbl.textContent = 'abcdefgh'[file];
          cell.appendChild(lbl);
        }

        cell.dataset.sq = idx;
        if (onSquareClick) cell.addEventListener('click', () => onSquareClick(idx, fen));
        container.appendChild(cell);
      }
    }
    return { board, turn };
  }

  // ── Training data editor ──────────────────────────────────────────────────
  api.registerTrainEditor('chess', function (root, network, nc) {
    const count = Array.isArray(network.trainingData && network.trainingData.samples)
      ? network.trainingData.samples.length : 0;

    root.innerHTML = `
      <div style="display:grid;gap:12px;">
        <p class="hint">Upload UCI game files to train the chess model. Each line should contain space-separated moves in UCI format (e.g. <code>e2e4 e7e5 g1f3 b8c6</code>). Both white and black moves become training examples.</p>
        <div style="background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:10px 14px;font-size:13px;">
          <span style="color:#aaa;">Loaded positions:</span>
          <span id="chess-pos-count" style="color:#7ec8e3;font-weight:bold;margin-left:6px;">${count.toLocaleString()}</span>
        </div>
        <div class="row" style="gap:8px;">
          <button class="btn primary" id="btn-chess-upload">Upload game files (.txt)</button>
          <button class="btn" id="btn-chess-clear">Clear training data</button>
        </div>
        <div id="chess-upload-status" style="font-size:13px;color:#8bc34a;min-height:18px;"></div>
        <div style="background:#111;border:1px solid #2a2a2a;border-radius:6px;padding:10px 14px;">
          <div style="font-size:11px;color:#666;margin-bottom:4px;">Example file format (one game per line):</div>
          <code style="font-size:11px;color:#aaa;white-space:pre;">e2e4 e7e5 g1f3 b8c6 f1b5 a7a6
d2d4 d7d5 c2c4 e7e6 b1c3 g8f6</code>
        </div>
      </div>
    `;

    document.getElementById('btn-chess-upload').addEventListener('click', async () => {
      const statusEl = document.getElementById('chess-upload-status');
      statusEl.textContent = 'Reading files…';
      try {
        const file = await window.nc.dialog.readTextFile({ filters: [{ name: 'Text/PGN', extensions: ['txt', 'pgn', 'csv'] }] });
        if (!file) { statusEl.textContent = ''; return; }
        statusEl.textContent = 'Parsing games…';
        const result = await nc.invoke('chess:parseGames', file.content);
        if (!result.samples.length) { statusEl.style.color = '#f44'; statusEl.textContent = 'No valid UCI games found in file.'; return; }
        await window.nc.networks.update(network.id, { trainingData: { samples: result.samples } });
        document.getElementById('chess-pos-count').textContent = result.positionCount.toLocaleString();
        statusEl.style.color = '#8bc34a';
        statusEl.textContent = `Loaded ${result.positionCount.toLocaleString()} positions from ${result.gameCount} game(s).`;
      } catch (e) {
        statusEl.style.color = '#f44336';
        statusEl.textContent = 'Error: ' + e.message;
      }
    });

    document.getElementById('btn-chess-clear').addEventListener('click', async () => {
      await window.nc.networks.update(network.id, { trainingData: { samples: [] } });
      document.getElementById('chess-pos-count').textContent = '0';
      document.getElementById('chess-upload-status').textContent = 'Training data cleared.';
    });
  });

  // ── Inference UI ──────────────────────────────────────────────────────────
  api.registerInferenceRenderer('chess', function (root, network, nc) {
    const isTrained = !!(network.state || network.stateLocked);

    // ── State ──
    let fen = START_FEN;
    let highlights = {};
    let selectedSq = null;
    let lastPrediction = null;
    let pseudoLegal = [];

    // mode: 'manual' | 'white' (user=white, AI=black) | 'black' (user=black, AI=white) | 'ai-vs-ai'
    let mode = 'manual';
    let aiRunning = false;
    let aiVsAiTimer = null;

    // ── HTML ──
    root.innerHTML = `
      <div class="panel">
        <h2>Chess Inference — <span id="ci-netname"></span></h2>
        <div style="display:grid;grid-template-columns:380px 1fr;gap:20px;align-items:start;">

          <!-- Board column -->
          <div>
            <div id="chess-board-wrap"></div>
            <div style="margin-top:8px;display:flex;gap:8px;align-items:center;">
              <span id="ci-turn-label" style="font-size:13px;color:#ccc;"></span>
              <div class="spacer"></div>
              <button class="btn sm" id="ci-btn-reset">Reset</button>
            </div>
            <div id="ci-game-status" style="margin-top:6px;font-size:13px;min-height:20px;color:#f0b429;font-weight:600;text-align:center;"></div>
          </div>

          <!-- Controls column -->
          <div style="display:grid;gap:14px;">

            <div class="section">
              <h3>Game Mode</h3>
              ${!isTrained ? `<div style="background:#2a1a00;border:1px solid #664400;border-radius:4px;padding:8px 12px;font-size:12px;color:#f0b429;margin-bottom:8px;">Train the network on the Train tab to enable AI-powered game modes.</div>` : ''}
              <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:8px;">
                <button class="btn sm primary" id="ci-btn-play-white" ${!isTrained ? 'disabled' : ''}>Play as White</button>
                <button class="btn sm primary" id="ci-btn-play-black" ${!isTrained ? 'disabled' : ''}>Play as Black</button>
                <button class="btn sm" id="ci-btn-ai-vs-ai" ${!isTrained ? 'disabled' : ''}>AI vs AI</button>
                <button class="btn sm danger" id="ci-btn-stop-mode" style="display:none;">Stop</button>
              </div>
              <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#888;">
                <span>AI vs AI delay:</span>
                <input id="ci-aivai-delay" type="number" min="0.05" max="30" step="0.05" value="0.2"
                  style="width:64px;background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:3px 6px;color:#e0e0e0;font-size:12px;">
                <span>seconds</span>
              </div>
              <div id="ci-mode-status" style="margin-top:6px;font-size:12px;color:#888;min-height:16px;">Manual mode — click pieces to move freely.</div>
            </div>

            <div class="section">
              <h3>Position (FEN)</h3>
              <input id="ci-fen" type="text" style="width:100%;background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:6px 8px;color:#e0e0e0;font-size:12px;font-family:monospace;" value="${START_FEN}">
              <div class="row" style="margin-top:6px;gap:6px;">
                <button class="btn sm primary" id="ci-btn-load-fen">Load FEN</button>
                <button class="btn sm" id="ci-btn-predict" ${!isTrained ? 'disabled' : ''}>Predict best move</button>
              </div>
            </div>

            <div class="section">
              <h3>Last AI Move</h3>
              <div id="ci-pred-output" style="font-size:13px;color:#aaa;min-height:60px;">—</div>
            </div>

            <div class="section">
              <h3>Top moves</h3>
              <div id="ci-top-moves" style="font-size:12px;color:#888;min-height:40px;">—</div>
            </div>

          </div>
        </div>
      </div>
    `;

    document.getElementById('ci-netname').textContent = network.name;

    // ── Helpers ──

    function el(id) { return document.getElementById(id); }

    function setGameStatus(msg, color) {
      const s = el('ci-game-status');
      s.textContent = msg;
      s.style.color = color || '#f0b429';
    }

    function setModeStatus(msg) {
      const s = el('ci-mode-status');
      if (s) s.textContent = msg;
    }

    function updateTurnLabel() {
      const { turn } = parseFENBoard(fen);
      el('ci-turn-label').textContent = turn === 'white' ? '⬜ White to move' : '⬛ Black to move';
    }

    function refreshBoard() {
      const clickable = (mode === 'manual') ||
        (mode === 'white' && parseFENBoard(fen).turn === 'white') ||
        (mode === 'black' && parseFENBoard(fen).turn === 'black');
      renderBoard(el('chess-board-wrap'), fen, highlights, clickable ? handleSquareClick : null);
    }

    function updateModeStatus() {
      if (mode === 'manual') {
        setModeStatus('Manual mode — click pieces to move, or use Predict best move.');
        return;
      }
      const { turn } = parseFENBoard(fen);
      if (mode === 'white') {
        setModeStatus(turn === 'white' ? 'Your turn — move a white piece.' : 'AI is thinking…');
      } else if (mode === 'black') {
        setModeStatus(turn === 'black' ? 'Your turn — move a black piece.' : 'AI is thinking…');
      } else if (mode === 'ai-vs-ai') {
        setModeStatus(`AI vs AI — ${turn} to move.`);
      }
    }

    function activateMode(newMode) {
      if (aiVsAiTimer) { clearTimeout(aiVsAiTimer); aiVsAiTimer = null; }
      mode = newMode;
      el('ci-btn-stop-mode').style.display = (newMode !== 'manual') ? '' : 'none';
      updateModeStatus();
      refreshBoard();
    }

    function stopMode() {
      if (aiVsAiTimer) { clearTimeout(aiVsAiTimer); aiVsAiTimer = null; }
      activateMode('manual');
    }

    // ── Game status handler ──

    // Returns true if the game has reached a terminal state (caller should stop).
    async function handleGameStatus() {
      try {
        const status = await nc.invoke('chess:gameStatus', fen);
        if (status === 'checkmate') {
          const matedColor = fen.split(' ')[1] === 'w' ? 'White' : 'Black';
          const winner     = matedColor === 'White' ? 'Black' : 'White';
          setGameStatus(`Checkmate! ${winner} wins.`, '#f44336');
          stopMode();
          return true;
        }
        if (status === 'stalemate') {
          setGameStatus("Stalemate — it's a draw!", '#aaaaaa');
          stopMode();
          return true;
        }
        if (status === 'draw-50') {
          setGameStatus('Draw by 50-move rule!', '#aaaaaa');
          stopMode();
          return true;
        }
        if (status === 'check') {
          setGameStatus('♚ Check!', '#f0b429');
        } else {
          setGameStatus('');
        }
      } catch (e) { /* status check failed — non-fatal */ }
      return false;
    }

    // ── AI move logic ──

    async function aiMove() {
      if (aiRunning) return false;
      aiRunning = true;
      let moved = false;
      try {
        updateModeStatus();
        const encoded = await nc.invoke('chess:encodePosition', fen);
        const result = await window.nc.inference.run(network.id, { input: encoded });
        if (!result || result.kind !== 'classification') {
          stopMode(); return false;
        }
        const top = await nc.invoke('chess:topLegalMoves', fen, result.probs);

        if (!top.length) {
          // No legal moves — handled by handleGameStatus after the previous move,
          // but catch it here as a safety net
          await handleGameStatus();
          el('ci-pred-output').textContent = 'No legal moves available.';
          el('ci-top-moves').textContent = '';
          return false;
        }

        const best = top[0];
        el('ci-pred-output').innerHTML =
          `<span style="color:#7ec8e3;font-size:15px;font-weight:bold;">${best.from} → ${best.to}</span>` +
          ` <span style="color:#666;">(${(best.prob * 100).toFixed(1)}%)</span>`;
        el('ci-top-moves').innerHTML = top.map((m, i) =>
          `<div style="color:${i===0?'#7ec8e3':'#666'};padding:1px 0;">${i+1}. ${m.from}→${m.to}` +
          ` <span style="color:#444;">${(m.prob*100).toFixed(2)}%</span></div>`
        ).join('');

        highlights = { [nameToSq(best.from)]: 'from', [nameToSq(best.to)]: 'to' };
        fen = await nc.invoke('chess:applyMove', fen, best.uci);
        el('ci-fen').value = fen;
        selectedSq = null; pseudoLegal = [];
        refreshBoard();
        updateTurnLabel();

        // Check game status after AI move — may end the game
        if (await handleGameStatus()) return false;

        moved = true;
      } finally {
        aiRunning = false;
      }
      return moved;
    }

    // Called after user makes a move in play modes — triggers AI response if needed
    async function onUserMove() {
      if (mode === 'manual') return;
      const { turn } = parseFENBoard(fen);
      const aiShouldMove =
        (mode === 'white' && turn === 'black') ||
        (mode === 'black' && turn === 'white');
      if (!aiShouldMove) return;
      updateModeStatus();
      await aiMove();
      updateModeStatus();
    }

    // AI vs AI: one step, schedules the next
    async function aiVsAiStep() {
      if (mode !== 'ai-vs-ai') return;
      const moved = await aiMove();
      if (!moved || mode !== 'ai-vs-ai') return;
      updateModeStatus();
      const delaySec = parseFloat(el('ci-aivai-delay').value) || 0.2;
      const delayMs  = Math.max(50, Math.round(delaySec * 1000));
      aiVsAiTimer = setTimeout(aiVsAiStep, delayMs);
    }

    // ── Manual predict (analysis tool) ──

    async function predict() {
      const predEl = el('ci-pred-output');
      const topEl  = el('ci-top-moves');
      predEl.textContent = 'Thinking…';
      topEl.textContent  = '';
      try {
        const encoded = await nc.invoke('chess:encodePosition', fen);
        const result  = await window.nc.inference.run(network.id, { input: encoded });
        if (!result || result.kind !== 'classification') {
          predEl.textContent = 'Unexpected result from model.'; return;
        }
        const top = await nc.invoke('chess:topLegalMoves', fen, result.probs);
        lastPrediction = top[0] || null;

        if (lastPrediction) {
          highlights = { [nameToSq(lastPrediction.from)]: 'from', [nameToSq(lastPrediction.to)]: 'to' };
          refreshBoard();
          predEl.innerHTML =
            `<span style="color:#7ec8e3;font-size:15px;font-weight:bold;">${lastPrediction.from} → ${lastPrediction.to}</span>` +
            ` <span style="color:#666;">(${(lastPrediction.prob * 100).toFixed(1)}%)</span>` +
            `<br><button class="btn sm" id="ci-btn-apply" style="margin-top:6px;">Apply this move</button>`;
          el('ci-btn-apply').addEventListener('click', applyPredicted);
        } else {
          predEl.textContent = 'No legal moves predicted.';
          highlights = {};
          refreshBoard();
        }

        topEl.innerHTML = top.map((m, i) =>
          `<div style="color:${i===0?'#7ec8e3':'#666'};padding:1px 0;">${i+1}. ${m.from}→${m.to}` +
          ` <span style="color:#444;">${(m.prob*100).toFixed(2)}%</span></div>`
        ).join('') || '<div>No legal moves found.</div>';

      } catch (e) {
        predEl.textContent = 'Error: ' + e.message;
      }
    }

    async function applyPredicted() {
      if (!lastPrediction) return;
      await applyUCI(lastPrediction.uci); // use full UCI so promotions include the piece char
    }

    // ── Move application ──

    async function applyUCI(uci) {
      try {
        fen = await nc.invoke('chess:applyMove', fen, uci);
        el('ci-fen').value = fen;
        selectedSq = null; highlights = {}; lastPrediction = null; pseudoLegal = [];
        refreshBoard();
        updateTurnLabel();
        // Check if the move created check / checkmate / stalemate
        if (await handleGameStatus()) return;
        updateModeStatus();
        await onUserMove();
      } catch (e) {
        console.error('applyUCI error:', e);
      }
    }

    // ── Board click handler ──

    async function handleSquareClick(sqIdx, currentFen) {
      if (aiRunning) return;
      const { board, turn } = parseFENBoard(currentFen);

      // Enforce turn restrictions in play modes
      if (mode === 'white' && turn !== 'white') return;
      if (mode === 'black' && turn !== 'black') return;
      if (mode === 'ai-vs-ai') return;

      if (selectedSq === null) {
        if (board[sqIdx] === 0) return;
        // In play modes, only allow selecting own pieces
        if (mode === 'white' && board[sqIdx] < 0) return;
        if (mode === 'black' && board[sqIdx] > 0) return;
        selectedSq = sqIdx;
        highlights = { [sqIdx]: 'sel' };
        try {
          const legalUci = await nc.invoke('chess:legalMoves', fen);
          const fromName = sqName(sqIdx);
          pseudoLegal = legalUci.filter(m => m.startsWith(fromName));
          for (const m of pseudoLegal) {
            const toIdx = nameToSq(m.slice(2, 4));
            if (!highlights[toIdx]) highlights[toIdx] = 'dest';
          }
        } catch (e) { /* ignore */ }
        refreshBoard();
      } else {
        const fromName = sqName(selectedSq);
        const toName   = sqName(sqIdx);
        const uci      = fromName + toName;
        // Promotion: multiple legal moves share the same from→to, differing only by promo piece
        const isPromo  = pseudoLegal.some(m => m.length === 5 && m.startsWith(uci));
        if (isPromo) {
          const isWhite = parseFENBoard(fen).board[selectedSq] > 0;
          const promo   = await showPromotionDialog(el('chess-board-wrap'), isWhite);
          await applyUCI(uci + promo);
        } else {
          const match = pseudoLegal.find(m => m.startsWith(uci));
          if (match) {
            await applyUCI(match);
          } else if (sqIdx === selectedSq) {
            selectedSq = null; highlights = {}; pseudoLegal = []; refreshBoard();
          } else {
            const { board: b2 } = parseFENBoard(fen);
            if (b2[sqIdx] !== 0) {
              // Check the new piece respects play mode
              if (mode === 'white' && b2[sqIdx] < 0) { selectedSq = null; highlights = {}; pseudoLegal = []; refreshBoard(); return; }
              if (mode === 'black' && b2[sqIdx] > 0) { selectedSq = null; highlights = {}; pseudoLegal = []; refreshBoard(); return; }
              selectedSq = null; highlights = {}; pseudoLegal = [];
              await handleSquareClick(sqIdx, fen);
            } else {
              selectedSq = null; highlights = {}; pseudoLegal = []; refreshBoard();
            }
          }
        }
      }
    }

    // ── Button wiring ──

    el('ci-btn-play-white').addEventListener('click', async () => {
      fen = START_FEN;
      el('ci-fen').value = fen;
      selectedSq = null; highlights = {}; lastPrediction = null; pseudoLegal = [];
      el('ci-pred-output').textContent = '—';
      el('ci-top-moves').textContent = '—';
      setGameStatus('');
      activateMode('white');
      updateTurnLabel();
      // White goes first — user plays, no immediate AI move needed
      setModeStatus('Your turn — move a white piece.');
    });

    el('ci-btn-play-black').addEventListener('click', async () => {
      fen = START_FEN;
      el('ci-fen').value = fen;
      selectedSq = null; highlights = {}; lastPrediction = null; pseudoLegal = [];
      el('ci-pred-output').textContent = '—';
      el('ci-top-moves').textContent = '—';
      setGameStatus('');
      activateMode('black');
      updateTurnLabel();
      // AI is white and goes first
      setModeStatus('AI is thinking…');
      await aiMove();
      updateModeStatus();
    });

    el('ci-btn-ai-vs-ai').addEventListener('click', async () => {
      selectedSq = null; highlights = {}; lastPrediction = null; pseudoLegal = [];
      el('ci-pred-output').textContent = '—';
      el('ci-top-moves').textContent = '—';
      setGameStatus('');
      activateMode('ai-vs-ai');
      updateTurnLabel();
      aiVsAiStep();
    });

    el('ci-btn-stop-mode').addEventListener('click', () => {
      stopMode();
    });

    el('ci-btn-reset').addEventListener('click', () => {
      if (aiVsAiTimer) { clearTimeout(aiVsAiTimer); aiVsAiTimer = null; }
      mode = 'manual';
      el('ci-btn-stop-mode').style.display = 'none';
      fen = START_FEN;
      el('ci-fen').value = fen;
      highlights = {}; selectedSq = null; lastPrediction = null; pseudoLegal = [];
      el('ci-pred-output').textContent = '—';
      el('ci-top-moves').textContent = '—';
      setGameStatus('');
      setModeStatus('Manual mode — click pieces to move, or use Predict best move.');
      refreshBoard();
      updateTurnLabel();
    });

    el('ci-btn-load-fen').addEventListener('click', () => {
      const input = el('ci-fen').value.trim();
      if (!input) return;
      if (aiVsAiTimer) { clearTimeout(aiVsAiTimer); aiVsAiTimer = null; }
      mode = 'manual';
      el('ci-btn-stop-mode').style.display = 'none';
      fen = input;
      highlights = {}; selectedSq = null; lastPrediction = null; pseudoLegal = [];
      setGameStatus('');
      setModeStatus('Manual mode — click pieces to move, or use Predict best move.');
      refreshBoard();
      updateTurnLabel();
    });

    el('ci-btn-predict').addEventListener('click', predict);

    // Initial render
    refreshBoard();
    updateTurnLabel();
  });

})(api); // api is injected by the plugin registry
