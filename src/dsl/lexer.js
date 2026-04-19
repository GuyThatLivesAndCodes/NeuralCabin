'use strict';

// NeuralScript lexer. Produces tokens for a small, friendly language.
//
// Token kinds:
//   NUMBER, STRING, IDENT, KEYWORD, PUNCT, NEWLINE, EOF

const KEYWORDS = new Set([
  'let', 'set', 'if', 'else', 'while', 'for', 'to', 'by', 'fn', 'return',
  'true', 'false', 'null', 'print', 'and', 'or', 'not', 'in', 'do', 'end'
]);

function tokenize(source) {
  const tokens = [];
  let i = 0;
  let line = 1, col = 1;

  function peek(n = 0) { return source[i + n]; }
  function advance() {
    const c = source[i++];
    if (c === '\n') { line++; col = 1; } else col++;
    return c;
  }
  function push(type, value, startLine, startCol) {
    tokens.push({ type, value, line: startLine, col: startCol });
  }

  while (i < source.length) {
    const c = source[i];
    const startLine = line, startCol = col;
    if (c === ' ' || c === '\t' || c === '\r') { advance(); continue; }
    if (c === '\n') {
      advance();
      // collapse runs of newlines
      if (tokens.length && tokens[tokens.length - 1].type !== 'NEWLINE') push('NEWLINE', '\n', startLine, startCol);
      continue;
    }
    if (c === '#' || (c === '/' && peek(1) === '/')) {
      // line comment
      while (i < source.length && source[i] !== '\n') advance();
      continue;
    }
    if (c === '"' || c === "'") {
      const quote = c; advance();
      let s = '';
      while (i < source.length && source[i] !== quote) {
        if (source[i] === '\\' && i + 1 < source.length) {
          advance();
          const esc = advance();
          s += ({ n: '\n', t: '\t', r: '\r', '\\': '\\', '"': '"', "'": "'" })[esc] ?? esc;
        } else {
          s += advance();
        }
      }
      if (source[i] !== quote) throw new Error(`Unterminated string at line ${startLine}`);
      advance();
      push('STRING', s, startLine, startCol);
      continue;
    }
    if (/[0-9]/.test(c) || (c === '-' && /[0-9]/.test(peek(1)) && (!tokens.length || tokens[tokens.length - 1].type === 'PUNCT' || tokens[tokens.length - 1].type === 'NEWLINE' || tokens[tokens.length - 1].type === 'KEYWORD'))) {
      let num = advance();
      while (i < source.length && /[0-9\._eE+\-]/.test(source[i])) {
        const ch = source[i];
        if ((ch === '+' || ch === '-') && num[num.length - 1] !== 'e' && num[num.length - 1] !== 'E') break;
        num += advance();
      }
      push('NUMBER', parseFloat(num), startLine, startCol);
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let ident = '';
      while (i < source.length && /[A-Za-z0-9_]/.test(source[i])) ident += advance();
      if (KEYWORDS.has(ident)) push('KEYWORD', ident, startLine, startCol);
      else push('IDENT', ident, startLine, startCol);
      continue;
    }
    // punctuation / operators
    const two = c + (peek(1) || '');
    if (['==', '!=', '<=', '>=', '->', '&&', '||'].includes(two)) {
      advance(); advance();
      push('PUNCT', two, startLine, startCol);
      continue;
    }
    if ('+-*/%=<>(){}[],;:.'.includes(c)) {
      advance();
      push('PUNCT', c, startLine, startCol);
      continue;
    }
    throw new Error(`Unexpected character ${JSON.stringify(c)} at line ${line} col ${col}`);
  }
  push('EOF', null, line, col);
  return tokens;
}

module.exports = { tokenize, KEYWORDS };
