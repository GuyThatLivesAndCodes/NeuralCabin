'use strict';

// Chat formatting for character-level language models.
//
// We accept several JSON shapes and flatten each conversation into a single
// text stream with role tags the model can learn:
//
//   <|user|>Hello, I need help<|end|><|assistant|>Of course, I'm here<|end|>
//
// During inference we wrap the user's prompt the same way and stop generation
// when we hit the assistant end tag.

const USER_OPEN = '<|user|>';
const ASSISTANT_OPEN = '<|assistant|>';
const SYSTEM_OPEN = '<|system|>';
const END = '<|end|>';

// Public so callers (tests / docs) can reference them.
const TAGS = { USER_OPEN, ASSISTANT_OPEN, SYSTEM_OPEN, END };

// Detect whether a sample is a chat-style record. Returns the normalized form
// or null if not a chat sample.
//   { user, assistant }                      → one turn
//   { messages: [{role, content}, ...] }     → multi-turn (OpenAI-style)
//   { conversation: [{user|assistant, ...}]} → multi-turn alternating
function normalizeChatSample(s) {
  if (!s || typeof s !== 'object') return null;
  if (typeof s.user === 'string' && typeof s.assistant === 'string') {
    const turns = [{ role: 'user', content: s.user }, { role: 'assistant', content: s.assistant }];
    if (typeof s.system === 'string') turns.unshift({ role: 'system', content: s.system });
    return turns;
  }
  if (Array.isArray(s.messages) && s.messages.every(m => m && typeof m.role === 'string' && typeof m.content === 'string')) {
    return s.messages.map(m => ({ role: m.role, content: m.content }));
  }
  if (Array.isArray(s.conversation)) {
    const out = [];
    for (const turn of s.conversation) {
      if (!turn || typeof turn !== 'object') continue;
      if (typeof turn.user === 'string') out.push({ role: 'user', content: turn.user });
      if (typeof turn.assistant === 'string') out.push({ role: 'assistant', content: turn.assistant });
      if (typeof turn.system === 'string') out.push({ role: 'system', content: turn.system });
    }
    return out.length ? out : null;
  }
  return null;
}

function renderTurns(turns) {
  let out = '';
  for (const t of turns) {
    const open = t.role === 'assistant' ? ASSISTANT_OPEN : t.role === 'system' ? SYSTEM_OPEN : USER_OPEN;
    out += open + t.content + END;
  }
  return out;
}

// Take training data (whatever shape) and return:
//   { text: string, isChat: boolean, perSampleCount }
// Falls back to the existing plaintext behavior for non-chat samples.
function buildCorpus(data) {
  if (!data) return { text: '', isChat: false, perSampleCount: 0 };
  // Direct text wins
  if (typeof data.text === 'string' && (!data.samples || data.samples.length === 0)) {
    return { text: data.text, isChat: false, perSampleCount: 0 };
  }
  if (!Array.isArray(data.samples)) {
    if (typeof data.text === 'string') return { text: data.text, isChat: false, perSampleCount: 0 };
    return { text: '', isChat: false, perSampleCount: 0 };
  }
  // Look at the first non-null sample to decide
  const chats = [];
  const plains = [];
  for (const s of data.samples) {
    const turns = normalizeChatSample(s);
    if (turns) chats.push(turns);
    else if (typeof s?.text === 'string') plains.push(s.text);
  }
  if (chats.length > 0) {
    // Render every conversation, separated by a blank line for breathing room.
    const text = chats.map(renderTurns).join('\n');
    return { text, isChat: true, perSampleCount: chats.length };
  }
  if (plains.length > 0) {
    return { text: plains.join('\n'), isChat: false, perSampleCount: plains.length };
  }
  return { text: '', isChat: false, perSampleCount: 0 };
}

// Wrap a user prompt for inference against a chat-trained model.
// The returned string ends right after <|assistant|> so the model continues
// from there.
function wrapPromptForChat(userPrompt, system) {
  let out = '';
  if (typeof system === 'string' && system.length > 0) out += SYSTEM_OPEN + system + END;
  out += USER_OPEN + userPrompt + END + ASSISTANT_OPEN;
  return out;
}

// Strip role tags + everything after the first END tag in the assistant span.
// Used when post-processing generated text.
function extractAssistantReply(generated) {
  // generated is whatever came after wrapPromptForChat — i.e. assistant content
  // possibly followed by <|end|> and more turns.
  const endIdx = generated.indexOf(END);
  if (endIdx === -1) return generated;
  return generated.slice(0, endIdx);
}

module.exports = { TAGS, normalizeChatSample, renderTurns, buildCorpus, wrapPromptForChat, extractAssistantReply };
