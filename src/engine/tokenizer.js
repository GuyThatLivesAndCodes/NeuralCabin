'use strict';

// Simple char-level tokenizer. Builds vocab from corpus.
class CharTokenizer {
  constructor(chars) {
    this.chars = chars || [];
    this.stoi = new Map();
    this.itos = new Map();
    for (let i = 0; i < this.chars.length; i++) {
      this.stoi.set(this.chars[i], i);
      this.itos.set(i, this.chars[i]);
    }
  }
  get vocabSize() { return this.chars.length; }

  static fromCorpus(text) {
    const set = new Set(text);
    const chars = Array.from(set).sort();
    return new CharTokenizer(chars);
  }

  encode(text) {
    const out = new Array(text.length);
    for (let i = 0; i < text.length; i++) {
      const id = this.stoi.get(text[i]);
      if (id == null) throw new Error(`char not in vocab: ${JSON.stringify(text[i])}`);
      out[i] = id;
    }
    return out;
  }

  encodeSafe(text, fallback = null) {
    const out = [];
    for (let i = 0; i < text.length; i++) {
      const id = this.stoi.get(text[i]);
      if (id == null) {
        if (fallback != null) out.push(fallback);
      } else out.push(id);
    }
    return out;
  }

  decode(ids) {
    let s = '';
    for (const id of ids) s += this.itos.get(id) || '';
    return s;
  }

  toJSON() { return { chars: this.chars }; }
  static fromJSON(o) { return new CharTokenizer(o.chars); }
}

module.exports = { CharTokenizer };
