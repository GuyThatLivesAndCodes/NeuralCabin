'use strict';

// NeuralScript parser — produces an AST. Grammar is indentation-free,
// using `do`/`end` blocks to keep the lexer simple.
//
//   program      := (stmt NEWLINE)*
//   stmt         := letDecl | setDecl | ifStmt | whileStmt | forStmt | fnDecl
//                 | returnStmt | printStmt | exprStmt
//   letDecl      := 'let' IDENT '=' expr
//   setDecl      := 'set' IDENT '=' expr
//   ifStmt       := 'if' expr 'do' block ( 'else' 'do' block )? 'end'
//   whileStmt    := 'while' expr 'do' block 'end'
//   forStmt      := 'for' IDENT '=' expr 'to' expr ( 'by' expr )? 'do' block 'end'
//   fnDecl       := 'fn' IDENT '(' params? ')' 'do' block 'end'
//   returnStmt   := 'return' expr?
//   printStmt    := 'print' expr
//   block        := (stmt NEWLINE)*
//   expr         := orExpr
//   orExpr       := andExpr ('or' andExpr)*
//   andExpr      := notExpr ('and' notExpr)*
//   notExpr      := 'not' notExpr | cmpExpr
//   cmpExpr      := addExpr (('=='|'!='|'<'|'<='|'>'|'>=') addExpr)*
//   addExpr      := mulExpr (('+'|'-') mulExpr)*
//   mulExpr      := unary (('*'|'/'|'%') unary)*
//   unary        := ('-' unary) | postfix
//   postfix      := primary (call | index | dot)*
//   primary      := NUMBER | STRING | 'true'|'false'|'null' | IDENT | '(' expr ')' | list | object
//   list         := '[' (expr (',' expr)*)? ']'
//   object       := '{' ( (STRING|IDENT) ':' expr (',' (STRING|IDENT) ':' expr)* )? '}'
//   call         := '(' (expr (',' expr)*)? ')'
//   index        := '[' expr ']'
//   dot          := '.' IDENT

function parse(tokens) {
  let i = 0;

  function peek(n = 0) { return tokens[i + n]; }
  function eat() { return tokens[i++]; }
  function match(type, value) {
    const t = peek();
    if (!t) return false;
    if (t.type !== type) return false;
    if (value !== undefined && t.value !== value) return false;
    return true;
  }
  function expect(type, value, what) {
    const t = peek();
    if (!t || t.type !== type || (value !== undefined && t.value !== value)) {
      throw new Error(`Parse error at line ${t?.line}: expected ${what || (value ?? type)}, got ${JSON.stringify(t)}`);
    }
    return eat();
  }
  function skipNewlines() { while (match('NEWLINE')) eat(); }

  function parseProgram() {
    const body = [];
    skipNewlines();
    while (!match('EOF')) {
      body.push(parseStmt());
      skipNewlines();
    }
    return { type: 'Program', body };
  }

  function parseStmt() {
    const t = peek();
    if (t.type === 'KEYWORD') {
      switch (t.value) {
        case 'let': return parseLet();
        case 'set': return parseSet();
        case 'if': return parseIf();
        case 'while': return parseWhile();
        case 'for': return parseFor();
        case 'fn': return parseFn();
        case 'return': return parseReturn();
        case 'print': return parsePrint();
      }
    }
    return { type: 'ExpressionStatement', expression: parseExpr(), line: t.line };
  }

  function parseLet() {
    const tok = eat(); // 'let'
    const name = expect('IDENT', undefined, 'identifier').value;
    expect('PUNCT', '=', '=');
    const value = parseExpr();
    return { type: 'Let', name, value, line: tok.line };
  }
  function parseSet() {
    const tok = eat();
    const name = expect('IDENT', undefined, 'identifier').value;
    // optional index/dot chain for set
    const chain = [];
    while (match('PUNCT', '[') || match('PUNCT', '.')) {
      if (match('PUNCT', '[')) {
        eat();
        const k = parseExpr();
        expect('PUNCT', ']');
        chain.push({ kind: 'index', key: k });
      } else {
        eat();
        chain.push({ kind: 'dot', key: expect('IDENT').value });
      }
    }
    expect('PUNCT', '=');
    const value = parseExpr();
    return { type: 'Set', name, chain, value, line: tok.line };
  }
  function parseIf() {
    const tok = eat();
    const cond = parseExpr();
    expect('KEYWORD', 'do');
    skipNewlines();
    const consequent = parseBlockUntil(['else', 'end']);
    let alternate = null;
    if (match('KEYWORD', 'else')) {
      eat();
      expect('KEYWORD', 'do');
      skipNewlines();
      alternate = parseBlockUntil(['end']);
    }
    expect('KEYWORD', 'end');
    return { type: 'If', cond, consequent, alternate, line: tok.line };
  }
  function parseWhile() {
    const tok = eat();
    const cond = parseExpr();
    expect('KEYWORD', 'do');
    skipNewlines();
    const body = parseBlockUntil(['end']);
    expect('KEYWORD', 'end');
    return { type: 'While', cond, body, line: tok.line };
  }
  function parseFor() {
    const tok = eat();
    const name = expect('IDENT').value;
    expect('PUNCT', '=');
    const from = parseExpr();
    expect('KEYWORD', 'to');
    const toExpr = parseExpr();
    let by = null;
    if (match('KEYWORD', 'by')) { eat(); by = parseExpr(); }
    expect('KEYWORD', 'do');
    skipNewlines();
    const body = parseBlockUntil(['end']);
    expect('KEYWORD', 'end');
    return { type: 'For', name, from, to: toExpr, by, body, line: tok.line };
  }
  function parseFn() {
    const tok = eat();
    const name = expect('IDENT').value;
    expect('PUNCT', '(');
    const params = [];
    if (!match('PUNCT', ')')) {
      params.push(expect('IDENT').value);
      while (match('PUNCT', ',')) { eat(); params.push(expect('IDENT').value); }
    }
    expect('PUNCT', ')');
    expect('KEYWORD', 'do');
    skipNewlines();
    const body = parseBlockUntil(['end']);
    expect('KEYWORD', 'end');
    return { type: 'FnDecl', name, params, body, line: tok.line };
  }
  function parseReturn() {
    const tok = eat();
    let value = null;
    if (!match('NEWLINE') && !match('EOF') && !match('KEYWORD', 'end')) value = parseExpr();
    return { type: 'Return', value, line: tok.line };
  }
  function parsePrint() {
    const tok = eat();
    const value = parseExpr();
    return { type: 'Print', value, line: tok.line };
  }

  function parseBlockUntil(enders) {
    const body = [];
    while (true) {
      skipNewlines();
      if (match('EOF')) throw new Error(`Unexpected EOF — expected one of: ${enders.join(', ')}`);
      if (match('KEYWORD') && enders.includes(peek().value)) break;
      body.push(parseStmt());
      skipNewlines();
    }
    return body;
  }

  // Expressions
  function parseExpr() { return parseOr(); }
  function parseOr() {
    let left = parseAnd();
    while (match('KEYWORD', 'or') || match('PUNCT', '||')) {
      const op = eat().value;
      left = { type: 'Logical', op: 'or', left, right: parseAnd() };
    }
    return left;
  }
  function parseAnd() {
    let left = parseNot();
    while (match('KEYWORD', 'and') || match('PUNCT', '&&')) {
      const op = eat().value;
      left = { type: 'Logical', op: 'and', left, right: parseNot() };
    }
    return left;
  }
  function parseNot() {
    if (match('KEYWORD', 'not')) { eat(); return { type: 'Unary', op: 'not', value: parseNot() }; }
    return parseCmp();
  }
  function parseCmp() {
    let left = parseAdd();
    while (match('PUNCT', '==') || match('PUNCT', '!=') || match('PUNCT', '<') || match('PUNCT', '<=') || match('PUNCT', '>') || match('PUNCT', '>=')) {
      const op = eat().value;
      left = { type: 'Binary', op, left, right: parseAdd() };
    }
    return left;
  }
  function parseAdd() {
    let left = parseMul();
    while (match('PUNCT', '+') || match('PUNCT', '-')) {
      const op = eat().value;
      left = { type: 'Binary', op, left, right: parseMul() };
    }
    return left;
  }
  function parseMul() {
    let left = parseUnary();
    while (match('PUNCT', '*') || match('PUNCT', '/') || match('PUNCT', '%')) {
      const op = eat().value;
      left = { type: 'Binary', op, left, right: parseUnary() };
    }
    return left;
  }
  function parseUnary() {
    if (match('PUNCT', '-')) { eat(); return { type: 'Unary', op: '-', value: parseUnary() }; }
    return parsePostfix();
  }
  function parsePostfix() {
    let node = parsePrimary();
    while (true) {
      if (match('PUNCT', '(')) {
        eat();
        const args = [];
        if (!match('PUNCT', ')')) {
          args.push(parseExpr());
          while (match('PUNCT', ',')) { eat(); args.push(parseExpr()); }
        }
        expect('PUNCT', ')');
        node = { type: 'Call', callee: node, args };
      } else if (match('PUNCT', '[')) {
        eat();
        const key = parseExpr();
        expect('PUNCT', ']');
        node = { type: 'Index', object: node, key };
      } else if (match('PUNCT', '.')) {
        eat();
        const name = expect('IDENT').value;
        node = { type: 'Dot', object: node, name };
      } else break;
    }
    return node;
  }
  function parsePrimary() {
    const t = peek();
    if (t.type === 'NUMBER') { eat(); return { type: 'Num', value: t.value }; }
    if (t.type === 'STRING') { eat(); return { type: 'Str', value: t.value }; }
    if (t.type === 'KEYWORD' && (t.value === 'true' || t.value === 'false' || t.value === 'null')) {
      eat();
      return { type: 'Literal', value: t.value === 'true' ? true : t.value === 'false' ? false : null };
    }
    if (t.type === 'IDENT') { eat(); return { type: 'Ident', name: t.value }; }
    if (match('PUNCT', '(')) { eat(); const e = parseExpr(); expect('PUNCT', ')'); return e; }
    if (match('PUNCT', '[')) {
      eat();
      const items = [];
      skipNewlines();
      if (!match('PUNCT', ']')) {
        items.push(parseExpr());
        while (match('PUNCT', ',')) { eat(); skipNewlines(); if (match('PUNCT', ']')) break; items.push(parseExpr()); }
      }
      skipNewlines();
      expect('PUNCT', ']');
      return { type: 'List', items };
    }
    if (match('PUNCT', '{')) {
      eat();
      const entries = [];
      skipNewlines();
      if (!match('PUNCT', '}')) {
        entries.push(parseObjectEntry());
        while (match('PUNCT', ',')) { eat(); skipNewlines(); if (match('PUNCT', '}')) break; entries.push(parseObjectEntry()); }
      }
      skipNewlines();
      expect('PUNCT', '}');
      return { type: 'Obj', entries };
    }
    throw new Error(`Unexpected token ${JSON.stringify(t)} at line ${t.line}`);
  }
  function parseObjectEntry() {
    let key;
    const t = peek();
    if (t.type === 'STRING') { eat(); key = t.value; }
    else if (t.type === 'IDENT') { eat(); key = t.value; }
    else throw new Error(`Expected object key at line ${t.line}`);
    expect('PUNCT', ':');
    return { key, value: parseExpr() };
  }

  return parseProgram();
}

module.exports = { parse };
