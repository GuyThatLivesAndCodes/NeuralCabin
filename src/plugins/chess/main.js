'use strict';

// ─── Piece constants ─────────────────────────────────────────────────────────
const PIECE = { P:1, N:2, B:3, R:4, Q:5, K:6, p:-1, n:-2, b:-3, r:-4, q:-5, k:-6 };

// ─── Square helpers ───────────────────────────────────────────────────────────
// Squares indexed 0–63: index = rank*8 + file  (a1=0, h1=7, a8=56, h8=63)
function sq(file, rank) { return rank * 8 + file; }
function sqFile(idx) { return idx % 8; }
function sqRank(idx) { return Math.floor(idx / 8); }
function sqName(idx) { return 'abcdefgh'[sqFile(idx)] + (sqRank(idx) + 1); }
function nameToSq(s) { return sq(s.charCodeAt(0) - 97, parseInt(s[1]) - 1); }

// ─── FEN parsing ─────────────────────────────────────────────────────────────
function parseFEN(fen) {
  const parts = fen.trim().split(/\s+/);
  const placement = parts[0];
  const turn   = (parts[1] || 'w') === 'w' ? 1 : -1;
  const castle  = parts[2] || '-';
  const ep      = parts[3] || '-';
  const half    = parseInt(parts[4] || '0') || 0;
  const full    = parseInt(parts[5] || '1') || 1;

  const board = new Array(64).fill(0);
  let rank = 7, file = 0;
  for (const c of placement) {
    if (c === '/') { rank--; file = 0; }
    else if (c >= '1' && c <= '8') { file += parseInt(c); }
    else { board[sq(file, rank)] = PIECE[c] || 0; file++; }
  }
  return {
    board, turn,
    castling: { K: castle.includes('K'), Q: castle.includes('Q'), k: castle.includes('k'), q: castle.includes('q') },
    ep: ep !== '-' ? nameToSq(ep) : -1,
    half, full
  };
}

// ─── FEN generation ───────────────────────────────────────────────────────────
const PIECE_CHAR = { 1:'P', 2:'N', 3:'B', 4:'R', 5:'Q', 6:'K', '-1':'p', '-2':'n', '-3':'b', '-4':'r', '-5':'q', '-6':'k' };
function boardToFEN(state) {
  const rows = [];
  for (let r = 7; r >= 0; r--) {
    let row = '', empty = 0;
    for (let f = 0; f < 8; f++) {
      const p = state.board[sq(f, r)];
      if (p === 0) { empty++; }
      else { if (empty) { row += empty; empty = 0; } row += PIECE_CHAR[String(p)] || '?'; }
    }
    if (empty) row += empty;
    rows.push(row);
  }
  const castleStr = [
    state.castling.K ? 'K' : '', state.castling.Q ? 'Q' : '',
    state.castling.k ? 'k' : '', state.castling.q ? 'q' : ''
  ].join('') || '-';
  return `${rows.join('/')} ${state.turn === 1 ? 'w' : 'b'} ${castleStr} ${state.ep >= 0 ? sqName(state.ep) : '-'} ${state.half} ${state.full}`;
}

// ─── Position encoding (73 floats) ───────────────────────────────────────────
function encodePosition(fen) {
  const s = parseFEN(fen);
  const v = new Array(73).fill(0);
  for (let i = 0; i < 64; i++) v[i] = s.board[i] / 6.0;
  v[64] = s.turn;
  v[65] = s.castling.K ? 1 : 0;
  v[66] = s.castling.Q ? 1 : 0;
  v[67] = s.castling.k ? 1 : 0;
  v[68] = s.castling.q ? 1 : 0;
  if (s.ep >= 0) { v[69] = sqFile(s.ep) / 7; v[70] = sqRank(s.ep) / 7; }
  v[71] = Math.min(s.half / 100, 1);
  v[72] = Math.min(s.full / 200, 1);
  return Array.from(v);
}

// ─── Move application ─────────────────────────────────────────────────────────
function applyMove(fen, uciMove) {
  const s = parseFEN(fen);
  const from = nameToSq(uciMove.slice(0, 2));
  const to   = nameToSq(uciMove.slice(2, 4));
  const promo = uciMove[4] ? PIECE[uciMove[4]] * (s.turn > 0 ? 1 : -1) : 0;
  const board = s.board.slice();
  const piece = board[from];

  // En passant capture
  if (Math.abs(piece) === 1 && to === s.ep) {
    board[sq(sqFile(to), sqRank(from))] = 0;
  }
  // Castling rook move
  if (Math.abs(piece) === 6 && Math.abs(sqFile(to) - sqFile(from)) === 2) {
    const r = sqRank(from);
    if (sqFile(to) === 6) { board[sq(7, r)] = 0; board[sq(5, r)] = s.turn * 4; } // kingside
    else                  { board[sq(0, r)] = 0; board[sq(3, r)] = s.turn * 4; } // queenside
  }

  board[to]   = promo || piece;
  board[from] = 0;

  let newEp = -1;
  if (Math.abs(piece) === 1 && Math.abs(sqRank(to) - sqRank(from)) === 2) {
    newEp = sq(sqFile(from), (sqRank(from) + sqRank(to)) >> 1);
  }

  const c = { ...s.castling };
  if (Math.abs(piece) === 6) { if (s.turn > 0) { c.K = false; c.Q = false; } else { c.k = false; c.q = false; } }
  if (from === sq(7, 0) || to === sq(7, 0)) c.K = false;
  if (from === sq(0, 0) || to === sq(0, 0)) c.Q = false;
  if (from === sq(7, 7) || to === sq(7, 7)) c.k = false;
  if (from === sq(0, 7) || to === sq(0, 7)) c.q = false;

  const newHalf = (Math.abs(piece) === 1 || s.board[to] !== 0) ? 0 : s.half + 1;
  const newFull = s.turn === -1 ? s.full + 1 : s.full;

  return boardToFEN({ board, turn: -s.turn, castling: c, ep: newEp, half: newHalf, full: newFull });
}

// ─── Pseudo-legal move generation ────────────────────────────────────────────
// Generates moves following piece movement rules only — does not check for
// leaving own king in check. Use legalMoves() for enforcement.
function pseudoLegalMoves(fen) {
  const s = parseFEN(fen);
  const moves = [];
  const own = s.turn;
  const opp = -s.turn;

  function push(f, t, promo) {
    moves.push(promo ? sqName(f) + sqName(t) + promo : sqName(f) + sqName(t));
  }

  function slide(f, dirs) {
    for (const [df, dr] of dirs) {
      let ff = sqFile(f) + df, fr = sqRank(f) + dr;
      while (ff >= 0 && ff < 8 && fr >= 0 && fr < 8) {
        const ti = sq(ff, fr);
        if (Math.sign(s.board[ti]) === own) break;
        push(f, ti);
        if (s.board[ti] !== 0) break;
        ff += df; fr += dr;
      }
    }
  }

  for (let i = 0; i < 64; i++) {
    const p = s.board[i];
    if (Math.sign(p) !== own) continue;
    const ap = Math.abs(p);
    const file = sqFile(i), rank = sqRank(i);

    if (ap === 1) {
      const dir = own; const startRank = own === 1 ? 1 : 6; const promoRank = own === 1 ? 6 : 1;
      const fwd = sq(file, rank + dir);
      if (s.board[fwd] === 0) {
        if (rank === promoRank) { for (const pr of ['q','r','b','n']) push(i, fwd, pr); }
        else {
          push(i, fwd);
          if (rank === startRank && s.board[sq(file, rank + 2 * dir)] === 0) push(i, sq(file, rank + 2 * dir));
        }
      }
      for (const df of [-1, 1]) {
        if (file + df < 0 || file + df > 7) continue;
        const cap = sq(file + df, rank + dir);
        if (Math.sign(s.board[cap]) === opp || cap === s.ep) {
          if (rank === promoRank) { for (const pr of ['q','r','b','n']) push(i, cap, pr); }
          else { push(i, cap); }
        }
      }
    } else if (ap === 2) {
      for (const [df, dr] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
        const nf = file + df, nr = rank + dr;
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        const ti = sq(nf, nr);
        if (Math.sign(s.board[ti]) !== own) push(i, ti);
      }
    } else if (ap === 3) { slide(i, [[-1,-1],[-1,1],[1,-1],[1,1]]); }
      else if (ap === 4) { slide(i, [[-1,0],[1,0],[0,-1],[0,1]]); }
      else if (ap === 5) { slide(i, [[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]]); }
      else if (ap === 6) {
        for (const [df, dr] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
          const nf = file + df, nr = rank + dr;
          if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
          const ti = sq(nf, nr);
          if (Math.sign(s.board[ti]) !== own) push(i, ti);
        }
        // Castling — square occupancy only; check restrictions enforced in legalMoves()
        if (own === 1) {
          if (s.castling.K && s.board[sq(5,0)] === 0 && s.board[sq(6,0)] === 0) push(i, sq(6,0));
          if (s.castling.Q && s.board[sq(3,0)] === 0 && s.board[sq(2,0)] === 0 && s.board[sq(1,0)] === 0) push(i, sq(2,0));
        } else {
          if (s.castling.k && s.board[sq(5,7)] === 0 && s.board[sq(6,7)] === 0) push(i, sq(6,7));
          if (s.castling.q && s.board[sq(3,7)] === 0 && s.board[sq(2,7)] === 0 && s.board[sq(1,7)] === 0) push(i, sq(2,7));
        }
      }
  }
  return moves;
}

// ─── Attack detection ─────────────────────────────────────────────────────────
// Is `target` square attacked by any piece of `byColor` (1=white, -1=black)?
function isAttackedBy(board, target, byColor) {
  const tf = sqFile(target), tr = sqRank(target);

  // Pawns attack diagonally forward; looking backwards from target
  for (const df of [-1, 1]) {
    const f = tf + df, r = tr - byColor;
    if (f >= 0 && f < 8 && r >= 0 && r < 8 && board[sq(f, r)] === byColor) return true;
  }

  // Knights
  for (const [df, dr] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
    const f = tf + df, r = tr + dr;
    if (f >= 0 && f < 8 && r >= 0 && r < 8 && board[sq(f, r)] === byColor * 2) return true;
  }

  // Bishop / Queen (diagonals)
  for (const [df, dr] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
    let f = tf + df, r = tr + dr;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      const p = board[sq(f, r)];
      if (p !== 0) { if (p === byColor * 3 || p === byColor * 5) return true; break; }
      f += df; r += dr;
    }
  }

  // Rook / Queen (straights)
  for (const [df, dr] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    let f = tf + df, r = tr + dr;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      const p = board[sq(f, r)];
      if (p !== 0) { if (p === byColor * 4 || p === byColor * 5) return true; break; }
      f += df; r += dr;
    }
  }

  // King
  for (const [df, dr] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
    const f = tf + df, r = tr + dr;
    if (f >= 0 && f < 8 && r >= 0 && r < 8 && board[sq(f, r)] === byColor * 6) return true;
  }

  return false;
}

function findKing(board, color) {
  for (let i = 0; i < 64; i++) if (board[i] === color * 6) return i;
  return -1;
}

function isInCheck(board, color) {
  const kSq = findKing(board, color);
  return kSq >= 0 && isAttackedBy(board, kSq, -color);
}

// ─── Legal move generation ────────────────────────────────────────────────────
// Filters pseudo-legal moves to those that do not leave own king in check.
// Also enforces: no castling while in check, no castling through an attacked square.
function legalMoves(fen) {
  const s = parseFEN(fen);
  const pseudo = pseudoLegalMoves(fen);
  const legal = [];
  const inCheckNow = isInCheck(s.board, s.turn);

  for (const uci of pseudo) {
    const from = nameToSq(uci.slice(0, 2));
    const to   = nameToSq(uci.slice(2, 4));

    // Castling extra restrictions
    if (Math.abs(s.board[from]) === 6 && Math.abs(to - from) === 2) {
      if (inCheckNow) continue; // can't castle while in check
      const stepFile = sqFile(to) > sqFile(from) ? sqFile(from) + 1 : sqFile(from) - 1;
      if (isAttackedBy(s.board, sq(stepFile, sqRank(from)), -s.turn)) continue; // can't castle through check
    }

    // Apply and verify own king is not in check
    try {
      const newBoard = parseFEN(applyMove(fen, uci)).board;
      if (!isInCheck(newBoard, s.turn)) legal.push(uci);
    } catch (e) { /* skip malformed */ }
  }
  return legal;
}

// ─── Game status ─────────────────────────────────────────────────────────────
// Returns status from the perspective of the player whose turn it is.
// 'playing' | 'check' | 'checkmate' | 'stalemate' | 'draw-50'
function gameStatus(fen) {
  const s = parseFEN(fen);
  const inCheck = isInCheck(s.board, s.turn);
  const moves   = legalMoves(fen);
  if (moves.length === 0) return inCheck ? 'checkmate' : 'stalemate';
  if (inCheck) return 'check';
  if (s.half >= 100) return 'draw-50';
  return 'playing';
}

// ─── UCI helpers ──────────────────────────────────────────────────────────────
function uciToClass(uci) {
  return nameToSq(uci.slice(0, 2)) * 64 + nameToSq(uci.slice(2, 4));
}

function classToUci(cls) {
  return sqName(Math.floor(cls / 64)) + sqName(cls % 64);
}

// ─── UCI game file parser ─────────────────────────────────────────────────────
function parseGames(text) {
  const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const samples = [];
  let gameCount = 0;

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const tokens = line.replace(/\d+\./g, '').trim().split(/\s+/).filter(Boolean);
    if (tokens.length < 2) continue;

    let fen = START;
    let ok = true;
    gameCount++;

    for (const tok of tokens) {
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(tok)) continue;
      try {
        const input = encodePosition(fen);
        const output = uciToClass(tok);
        samples.push({ input, output });
        fen = applyMove(fen, tok);
      } catch (e) { ok = false; break; }
    }
    if (!ok) gameCount--;
  }

  return { samples, gameCount, positionCount: samples.length };
}

// ─── Top moves filtered to legal ─────────────────────────────────────────────
function topLegalMoves(fen, probs) {
  const legalList    = legalMoves(fen);
  const legalClasses = new Set(legalList.map(uciToClass));
  const scored = [];
  for (let i = 0; i < probs.length; i++) {
    if (legalClasses.has(i)) scored.push({ cls: i, prob: probs[i] });
  }
  scored.sort((a, b) => b.prob - a.prob);
  return scored.slice(0, 5).map(m => {
    const fromSq   = Math.floor(m.cls / 64);
    const toSq     = m.cls % 64;
    const fromName = sqName(fromSq);
    const toName   = sqName(toSq);
    // For promotions the neural net predicts the from→to class; find the actual UCI
    // from the legal list, preferring queen promotion (or any non-promotion exact match).
    let finalUci = legalList.find(l => l.startsWith(fromName + toName) && (l.endsWith('q') || l.length === 4));
    if (!finalUci) finalUci = fromName + toName; // fallback (shouldn't happen for legal moves)
    return { uci: finalUci, prob: m.prob, from: fromName, to: toName };
  });
}

// ─── Plugin export ────────────────────────────────────────────────────────────
module.exports = {
  mainHandlers: {
    'chess:encodePosition': (_, fen)        => encodePosition(fen),
    'chess:parseGames':     (_, text)       => parseGames(text),
    'chess:applyMove':      (_, fen, uci)   => applyMove(fen, uci),
    'chess:topLegalMoves':  (_, fen, probs) => topLegalMoves(fen, probs),
    'chess:legalMoves':     (_, fen)        => legalMoves(fen),
    'chess:gameStatus':     (_, fen)        => gameStatus(fen)
  }
};
