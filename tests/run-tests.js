'use strict';

// Engine self-test harness. Runs quick sanity checks on tensor ops,
// autograd, MLP training, and the DSL.

const T = require('../src/engine/tensor');
const { buildFromSpec } = require('../src/engine/model');
const { buildOptim } = require('../src/engine/optim');
const { trainNetwork, infer } = require('../src/engine/trainer');
const { tokenize } = require('../src/dsl/lexer');
const { parse } = require('../src/dsl/parser');
const { runScript } = require('../src/dsl/interpreter');

let passed = 0, failed = 0;

function check(name, cond, detail) {
  if (cond) { passed++; console.log('  OK   ', name); }
  else { failed++; console.log('  FAIL ', name, detail || ''); }
}

async function testTensor() {
  console.log('\n== tensor & autograd ==');
  // matmul
  const a = T.tensor([2, 3], [1, 2, 3, 4, 5, 6], true);
  const b = T.tensor([3, 2], [7, 8, 9, 10, 11, 12], true);
  const c = T.matmul(a, b);
  // expected: [[1*7+2*9+3*11, 1*8+2*10+3*12],[4*7+5*9+6*11, 4*8+5*10+6*12]]
  // = [[58, 64], [139, 154]]
  check('matmul values', c.data[0] === 58 && c.data[1] === 64 && c.data[2] === 139 && c.data[3] === 154);
  const loss = T.sumAll(c);
  loss.backward();
  // dA = ones * B^T rows summed; expect a.grad[0,0] = sum(B row 0) = 15
  check('matmul grad a', Math.abs(a.grad[0] - 15) < 1e-6, 'got ' + a.grad[0]);

  // MSE
  const x = T.tensor([2, 2], [1, 2, 3, 4], true);
  const y = T.tensor([2, 2], [1, 1, 1, 1]);
  const m = T.mseLoss(x, y);
  // mean of [0,1,4,9] = 3.5
  check('mse value', Math.abs(m.data[0] - 3.5) < 1e-5);
  m.backward();
  // d/dx_i = 2/N * (x-y); for x[1]=2 y[1]=1: 2/4 * 1 = 0.5
  check('mse grad', Math.abs(x.grad[1] - 0.5) < 1e-5);

  // Softmax cross-entropy sanity
  const logits = T.tensor([2, 3], [1, 2, 3, 1, 1, 1], true);
  const ce = T.softmaxCrossEntropy(logits, [2, 0]);
  check('ce finite', isFinite(ce.data[0]));
  ce.backward();
  check('ce grad shape', logits.grad && logits.grad.length === 6);
}

async function testMLP() {
  console.log('\n== MLP trains on XOR ==');
  const net = {
    architecture: { kind: 'classifier', inputDim: 2, outputDim: 2, hidden: [8, 8], activation: 'relu', dropout: 0, classes: ['0', '1'] },
    training: { optimizer: 'adam', learningRate: 0.05, batchSize: 4, epochs: 300, seed: 42 },
    trainingData: { samples: [
      { input: [0,0], label: 0 }, { input: [0,1], label: 1 },
      { input: [1,0], label: 1 }, { input: [1,1], label: 0 }
    ] }
  };
  let last = null;
  const r = await trainNetwork(net, {
    onProgress: p => { last = p; }
  });
  check('XOR training final loss < 0.2', last && last.loss < 0.2, 'final=' + (last?.loss));
  const trained = { ...net, state: r.state };
  const r00 = infer(trained, [0, 0]);
  const r01 = infer(trained, [0, 1]);
  const r10 = infer(trained, [1, 0]);
  const r11 = infer(trained, [1, 1]);
  const ok = r00.predictedClass === 0 && r01.predictedClass === 1 && r10.predictedClass === 1 && r11.predictedClass === 0;
  check('XOR inference all correct', ok, JSON.stringify({ r00: r00.predictedClass, r01: r01.predictedClass, r10: r10.predictedClass, r11: r11.predictedClass }));
}

async function testCharLM() {
  console.log('\n== charLM trains ==');
  const net = {
    architecture: { kind: 'charLM', vocabSize: 0, embDim: 8, contextLen: 8, hidden: [32], activation: 'gelu', dropout: 0 },
    training: { optimizer: 'adam', learningRate: 0.01, batchSize: 16, epochs: 5, seed: 42 },
    trainingData: { text: 'hello world. neural networks learn patterns.\n'.repeat(50) }
  };
  let last = null;
  const r = await trainNetwork(net, { onProgress: p => last = p });
  check('charLM produced state', !!r.state);
  check('charLM produced tokenizer', !!r.tokenizer);
  check('charLM loss finite', last && isFinite(last.loss));
  const trained = { ...net, state: r.state, tokenizer: r.tokenizer };
  const out = infer(trained, { prompt: 'hello ', maxTokens: 20, temperature: 0.8 });
  check('charLM generation non-empty', typeof out.text === 'string' && out.text.length > 0, JSON.stringify(out));
}

async function testDSL() {
  console.log('\n== DSL ==');
  const r1 = await runScript(`
    let x = 2
    let y = 3
    print x + y
    print "hello " + "world"
    for i = 0 to 4 do
      print i
    end
    fn square(n) do
      return n * n
    end
    print square(7)
  `);
  check('DSL basic execution', r1.ok, r1.error);
  check('DSL output lines', r1.output.split('\n').length >= 8);

  const r2 = await runScript(`
    let spec = { kind: "classifier", inputDim: 2, outputDim: 2, hidden: [8], activation: "relu", classes: ["0","1"] }
    let data = { samples: [
      { input: [0,0], label: 0 }, { input: [0,1], label: 1 },
      { input: [1,0], label: 1 }, { input: [1,1], label: 0 }
    ] }
    let opts = { optimizer: "adam", learningRate: 0.05, batchSize: 4, epochs: 150, seed: 42 }
    let r = await(train(spec, data, opts))
    print r.metrics[len(r.metrics) - 1].loss
  `);
  check('DSL trains inline', r2.ok, r2.error);
}

async function testChatFormat() {
  console.log('\n== chat-format loader ==');
  const ChatFormat = require('../src/engine/chat-format');

  const a = ChatFormat.normalizeChatSample({ user: 'hi', assistant: 'hello' });
  check('chat: pair shape detected', Array.isArray(a) && a.length === 2 && a[0].role === 'user' && a[1].role === 'assistant');

  const b = ChatFormat.normalizeChatSample({ messages: [{ role: 'user', content: 'x' }, { role: 'assistant', content: 'y' }] });
  check('chat: messages shape detected', Array.isArray(b) && b.length === 2);

  const c = ChatFormat.normalizeChatSample({ conversation: [{ user: 'a' }, { assistant: 'b' }] });
  check('chat: conversation shape detected', Array.isArray(c) && c.length === 2);

  const d = ChatFormat.normalizeChatSample({ text: 'just text' });
  check('chat: plain text returns null', d === null);

  const corpus = ChatFormat.buildCorpus({ samples: [
    { user: 'Hello, I need help', assistant: "Of course, I'm here, what's up?" },
    { user: 'Hi',                 assistant: 'Hey!' }
  ] });
  check('chat: corpus marked isChat', corpus.isChat === true);
  check('chat: corpus contains role tags', corpus.text.includes('<|user|>') && corpus.text.includes('<|assistant|>') && corpus.text.includes('<|end|>'));
  check('chat: corpus per-sample count', corpus.perSampleCount === 2);

  const wrapped = ChatFormat.wrapPromptForChat('Hello, I need help', 'Be brief.');
  check('chat: wrapPromptForChat order', wrapped.startsWith('<|system|>Be brief.<|end|><|user|>Hello, I need help<|end|><|assistant|>'));

  const reply = ChatFormat.extractAssistantReply("Of course, I'm here, what's up?<|end|><|user|>...");
  check('chat: extractAssistantReply strips end+rest', reply === "Of course, I'm here, what's up?");
}

async function testChatTrainingEndToEnd() {
  console.log('\n== chat training end-to-end ==');
  const samples = [];
  const seed = [
    { user: 'Hello, I need help', assistant: "Of course, I'm here, what's up?" },
    { user: 'Hi',                 assistant: 'Hey! How can I help today?' },
    { user: 'Goodbye',            assistant: 'Take care!' }
  ];
  for (let i = 0; i < 30; i++) for (const s of seed) samples.push(s);
  const net = {
    architecture: { kind: 'charLM', vocabSize: 0, embDim: 16, contextLen: 24, hidden: [48], activation: 'gelu', dropout: 0 },
    training: { optimizer: 'adam', learningRate: 0.005, batchSize: 16, epochs: 3, seed: 42 },
    trainingData: { samples }
  };
  const r = await trainNetwork(net, {});
  check('chat: training produced state', !!r.state);
  check('chat: training produced tokenizer', !!r.tokenizer);
  // After training, isChat should now be set on the architecture
  check('chat: arch.isChat auto-set after training', net.architecture.isChat === true);

  const trained = { ...net, state: r.state, tokenizer: r.tokenizer };
  const out = infer(trained, { prompt: 'Hello, I need help', maxTokens: 60, temperature: 0.7 });
  check('chat: inference returns text', typeof out.text === 'string' && out.text.length >= 0);
  check('chat: inference flagged as chat', out.chat === true);
  check('chat: inference text has no role tags', !out.text.includes('<|user|>') && !out.text.includes('<|assistant|>') && !out.text.includes('<|end|>'));
}

async function testContinueTrainingPreservesState() {
  console.log('\n== continue training preserves weights + optimizer state ==');
  // Train an MLP for a short run, then continue from saved state and verify
  // (a) loss does NOT spike on the first step of the continuation (which would
  //     happen if the Adam moments were reset), and (b) weights actually carry
  //     over (final loss after continue is <= the loss we left off at, within
  //     a small tolerance for stochastic noise).
  const baseNet = {
    architecture: { kind: 'classifier', inputDim: 2, outputDim: 2, hidden: [8, 8], activation: 'relu', dropout: 0, classes: ['0', '1'] },
    training: { optimizer: 'adam', learningRate: 0.05, batchSize: 4, epochs: 100, seed: 42 },
    trainingData: { samples: [
      { input: [0,0], label: 0 }, { input: [0,1], label: 1 },
      { input: [1,0], label: 1 }, { input: [1,1], label: 0 }
    ] }
  };
  let lastA = null;
  const r1 = await trainNetwork(baseNet, { onProgress: p => lastA = p });
  check('phase1: produced state', !!r1.state);
  check('phase1: produced optimizerState', !!r1.optimizerState && r1.optimizerState.type === 'adam');
  check('phase1: optimizer step counter advanced', r1.optimizerState.t > 0, 't=' + r1.optimizerState?.t);

  // Continue: same net carrying state + optimizerState forward.
  const cont = { ...baseNet, state: r1.state, optimizerState: r1.optimizerState,
                 training: { ...baseNet.training, epochs: 50 } };
  let firstStepLoss = null, lastB = null;
  const r2 = await trainNetwork(cont, { onProgress: p => {
    if (firstStepLoss === null) firstStepLoss = p.loss;
    lastB = p;
  } });
  // The first-step loss after continuation should be close to where we left off
  // — not a spike. If Adam state were reset, we'd see a big bump (often 2-10x).
  // Allow a generous 2x margin to keep the test robust to stochastic batch picks.
  const baselineEnd = lastA.loss;
  check('continue: no first-step loss spike',
        firstStepLoss !== null && firstStepLoss < baselineEnd * 3 + 0.05,
        `firstStep=${firstStepLoss?.toFixed(4)} baselineEnd=${baselineEnd?.toFixed(4)}`);
  // Continued training should not be worse than where we left off (with a tiny
  // tolerance for tail noise on a tiny task).
  check('continue: final loss <= previous final + slack',
        lastB.loss <= baselineEnd + 0.05,
        `prev=${baselineEnd?.toFixed(4)} cont=${lastB.loss?.toFixed(4)}`);

  // fromScratch flag should ignore the saved state and start over.
  const cont2 = { ...baseNet, state: r1.state, optimizerState: r1.optimizerState,
                  training: { ...baseNet.training, epochs: 1 } };
  let firstStepScratch = null;
  await trainNetwork(cont2, { fromScratch: true, onProgress: p => {
    if (firstStepScratch === null) firstStepScratch = p.loss;
  } });
  // A fresh init on this XOR task starts well above the converged loss.
  // Asserting it's at least somewhat higher than baselineEnd catches the case
  // where fromScratch silently kept the trained weights.
  check('fromScratch: first-step loss is meaningfully higher than converged loss',
        firstStepScratch !== null && firstStepScratch > baselineEnd + 0.1,
        `scratch=${firstStepScratch?.toFixed(4)} converged=${baselineEnd?.toFixed(4)}`);
}

async function testOptimizerStateRoundtrip() {
  console.log('\n== optimizer state JSON roundtrip ==');
  // Build a tiny model + Adam, run a few steps, snapshot, build a fresh Adam,
  // load the snapshot, and verify the internal buffers match exactly.
  const { buildFromSpec } = require('../src/engine/model');
  const rng = T.rngFromSeed(123);
  const m = buildFromSpec({ kind: 'classifier', inputDim: 3, outputDim: 2, hidden: [4], activation: 'relu', classes: ['a','b'] }, rng);
  const opt = buildOptim('adam', m.params, { lr: 0.01 });
  // simulate a couple of grad steps
  for (let step = 0; step < 3; step++) {
    for (const p of m.params) {
      if (!p.grad) p.grad = new Float32Array(p.size);
      for (let i = 0; i < p.size; i++) p.grad[i] = (i % 2 === 0 ? 0.1 : -0.05) * (step + 1);
    }
    opt.step();
  }
  const snap = opt.toJSON();
  check('adam snap: type tag', snap.type === 'adam');
  check('adam snap: t advanced', snap.t === 3);

  const opt2 = buildOptim('adam', m.params, { lr: 0.01 });
  const ok = opt2.loadFromJSON(snap);
  check('adam load: returned true', ok === true);
  check('adam load: t restored', opt2.t === 3);
  // verify a buffer slot matches
  check('adam load: m[0] preserved', Math.abs(opt.m[0][0] - opt2.m[0][0]) < 1e-12, `${opt.m[0][0]} vs ${opt2.m[0][0]}`);
  check('adam load: v[0] preserved', Math.abs(opt.v[0][0] - opt2.v[0][0]) < 1e-12);

  // Mismatched param-shape snapshot should be rejected (arch change scenario).
  const mWrong = buildFromSpec({ kind: 'classifier', inputDim: 5, outputDim: 2, hidden: [4], activation: 'relu', classes: ['a','b'] }, rng);
  const optWrong = buildOptim('adam', mWrong.params, { lr: 0.01 });
  check('adam load: rejects shape mismatch', optWrong.loadFromJSON(snap) === false);
}

async function testMultiTurnChatFormat() {
  console.log('\n== multi-turn chat format ==');
  const ChatFormat = require('../src/engine/chat-format');

  // wrapHistoryForChat with empty history + a userPrompt should match wrapPromptForChat.
  const w1 = ChatFormat.wrapHistoryForChat([], { userPrompt: 'Hi', system: 'Be concise.' });
  const w2 = ChatFormat.wrapPromptForChat('Hi', 'Be concise.');
  check('chat-history: empty history matches wrapPromptForChat', w1 === w2);

  // Multi-turn: produces a properly tagged stream that ends with the assistant-open tag.
  const history = [
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello!' },
    { role: 'user', content: 'How are you?' },
    { role: 'assistant', content: 'Doing well, thanks.' }
  ];
  const wrapped = ChatFormat.wrapHistoryForChat(history, { userPrompt: 'What can you do?', system: 'Be concise.' });
  check('chat-history: starts with system anchor', wrapped.startsWith('<|system|>Be concise.<|end|>'));
  check('chat-history: contains all user turns',
        wrapped.includes('<|user|>Hi<|end|>') && wrapped.includes('<|user|>How are you?<|end|>') && wrapped.includes('<|user|>What can you do?<|end|>'));
  check('chat-history: contains assistant turns',
        wrapped.includes('<|assistant|>Hello!<|end|>') && wrapped.includes('<|assistant|>Doing well, thanks.<|end|>'));
  check('chat-history: ends with assistant-open (model continues)',
        wrapped.endsWith('<|assistant|>'));
  check('chat-history: assistant-open appears once at the end (no trailing END)',
        !wrapped.endsWith('<|end|>'));

  // Inline system message in history is hoisted to the front.
  const w3 = ChatFormat.wrapHistoryForChat([
    { role: 'user', content: 'a' },
    { role: 'system', content: 'inline-system' },
    { role: 'assistant', content: 'b' }
  ], {});
  check('chat-history: inline system hoisted to front', w3.startsWith('<|system|>inline-system<|end|>'));

  // truncateWrappedToFit: drops oldest non-system turns first, keeps system + trailing assistant-open.
  const charEncoder = (s) => Array.from(s); // 1 char = 1 token for testing
  const longHistory = [];
  for (let i = 0; i < 20; i++) {
    longHistory.push({ role: 'user', content: `q${i}` });
    longHistory.push({ role: 'assistant', content: `a${i}` });
  }
  const longWrapped = ChatFormat.wrapHistoryForChat(longHistory, { userPrompt: 'final', system: 'sys' });
  const truncated = ChatFormat.truncateWrappedToFit(longWrapped, charEncoder, 200);
  check('truncate: result fits maxLen', charEncoder(truncated).length <= 200,
        `len=${charEncoder(truncated).length}`);
  check('truncate: keeps system anchor', truncated.startsWith('<|system|>sys<|end|>'));
  check('truncate: keeps trailing assistant-open', truncated.endsWith('<|assistant|>'));
  check('truncate: keeps the most recent user turn ("final")',
        truncated.includes('<|user|>final<|end|>'));
  check('truncate: drops oldest turns (q0 gone)', !truncated.includes('<|user|>q0<|end|>'));
}

async function testMultiTurnInferenceEndToEnd() {
  console.log('\n== multi-turn inference end-to-end ==');
  // Train a small chat model on multi-turn conversations, then verify
  // infer() accepts {history, prompt} and returns a string reply.
  const ChatFormat = require('../src/engine/chat-format');
  const samples = [];
  // Multi-turn samples (using `messages` shape).
  const seedConvos = [
    [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'How are you?' },
      { role: 'assistant', content: 'I am well, thanks.' }
    ],
    [
      { role: 'user', content: 'Tell me a fact' },
      { role: 'assistant', content: 'The sky is blue.' },
      { role: 'user', content: 'Another?' },
      { role: 'assistant', content: 'Water is wet.' }
    ]
  ];
  for (let i = 0; i < 20; i++) for (const c of seedConvos) samples.push({ messages: c });

  // Sanity: chat-format should treat these as chat samples.
  const corpus = ChatFormat.buildCorpus({ samples });
  check('e2e: corpus marked isChat', corpus.isChat === true);
  check('e2e: corpus contains multiple turns per sample',
        (corpus.text.match(/<\|user\|>/g) || []).length >= 4);

  const net = {
    architecture: { kind: 'charLM', vocabSize: 0, embDim: 16, contextLen: 64, hidden: [48], activation: 'gelu', dropout: 0 },
    training: { optimizer: 'adam', learningRate: 0.005, batchSize: 16, epochs: 3, seed: 42 },
    trainingData: { samples }
  };
  const r = await trainNetwork(net, {});
  const trained = { ...net, state: r.state, tokenizer: r.tokenizer };

  // Single-turn inference still works (back-compat).
  const out1 = infer(trained, { prompt: 'Hi', maxTokens: 30, temperature: 0.6 });
  check('e2e: single-turn infer still returns text', typeof out1.text === 'string');
  check('e2e: single-turn flagged as chat', out1.chat === true);

  // Multi-turn: pass running history + new prompt.
  const out2 = infer(trained, {
    history: [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' }
    ],
    prompt: 'How are you?',
    maxTokens: 30,
    temperature: 0.6
  });
  check('e2e: multi-turn infer returns string', typeof out2.text === 'string');
  check('e2e: multi-turn output has no role tags',
        !out2.text.includes('<|user|>') && !out2.text.includes('<|assistant|>') && !out2.text.includes('<|end|>'));

  // `messages` alias works.
  const out3 = infer(trained, {
    messages: [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'How are you?' }
    ],
    maxTokens: 30,
    temperature: 0.6
  });
  check('e2e: messages alias works', typeof out3.text === 'string');

  // History longer than contextLen should still produce a result (truncation kicks in).
  const longHistory = [];
  for (let i = 0; i < 30; i++) {
    longHistory.push({ role: 'user', content: 'a question that is reasonably long ' + i });
    longHistory.push({ role: 'assistant', content: 'a reply that is also reasonably long ' + i });
  }
  let longOk = false, longErr = null;
  try {
    const out4 = infer(trained, { history: longHistory, prompt: 'final question', maxTokens: 10, temperature: 0.6 });
    longOk = typeof out4.text === 'string';
  } catch (e) { longErr = e.message; }
  check('e2e: long history truncates without error', longOk, longErr);
}

async function testInferencePaddingChoice() {
  console.log('\n== inference padding uses safe token (no <<<<<<-prefix garbage) ==');
  // Regression: when the chat prompt is shorter than contextLen, the trainer
  // used to left-pad with the first char of the prompt — which for chat mode
  // meant a long run of '<' before '<|user|>', a pattern the model never saw
  // in training. Resulted in tag-fragment garbage in the reply. The fix pads
  // with '\n' (the conversation separator at training time) instead.
  const ChatFormat = require('../src/engine/chat-format');
  const samples = [];
  const seed = [
    { user: 'Hello, I need help', assistant: "Of course, I am here." },
    { user: 'Hi', assistant: 'Hey!' },
    { user: 'Thanks', assistant: 'Welcome!' }
  ];
  for (let i = 0; i < 30; i++) for (const s of seed) samples.push(s);

  // Use a long contextLen so padding dominates a short prompt — this is the
  // exact scenario in which the bug manifests.
  const net = {
    architecture: { kind: 'charLM', vocabSize: 0, embDim: 16, contextLen: 96, hidden: [48], activation: 'gelu', dropout: 0 },
    training: { optimizer: 'adam', learningRate: 0.005, batchSize: 16, epochs: 4, seed: 42 },
    trainingData: { samples }
  };
  const r = await trainNetwork(net, {});
  const trained = { ...net, state: r.state, tokenizer: r.tokenizer };

  // Sanity: tokenizer learned '\n' (the trainer joins conversations with it).
  check('padding: tokenizer includes "\\n" (the training-time separator)',
        r.tokenizer.chars.includes('\n'),
        'chars=' + JSON.stringify(r.tokenizer.chars.slice(0, 30)));

  // Run inference with a SHORT prompt — most of the context will be padding.
  const out = infer(trained, { prompt: 'Hi', maxTokens: 40, temperature: 0.7 });
  check('padding: returns string', typeof out.text === 'string');
  // We can't assert "good content" deterministically on a tiny model, but we
  // CAN assert the model didn't degenerate into outputting role-tag fragments
  // (which is what the buggy padding caused). Counting tag fragments in the
  // visible output (post-extraction) is a tight proxy for the bug.
  const tagFragments = (out.text.match(/<\|/g) || []).length + (out.text.match(/\|>/g) || []).length;
  check('padding: output has no leaked role-tag fragments',
        tagFragments === 0,
        'text=' + JSON.stringify(out.text) + ' tagFragments=' + tagFragments);
}

async function testCharLMContinuePreservesWeights() {
  console.log('\n== charLM Continue training preserves weights (regression: loss spike) ==');
  // Reproduces a real bug: charLM trains, the trainer mutates arch.vocabSize
  // from 0 to N during phase 1 but storage didn't persist the mutation. On
  // continue-training, the trainer saw arch.vocabSize=0 and state.arch.vocabSize=N,
  // detected a "mismatch", and wiped the freshly-restored model — sending loss
  // back up to fresh-init levels.

  const arch = { kind: 'charLM', vocabSize: 0, embDim: 16, contextLen: 32, hidden: [48], activation: 'gelu', dropout: 0 };
  const baseTraining = { optimizer: 'adam', learningRate: 0.005, batchSize: 16, epochs: 4, seed: 42 };
  const text = 'The quick brown fox jumps over the lazy dog. ' .repeat(40);

  // Phase 1: initial training. arch.vocabSize starts at 0 and the trainer fills it in.
  const phase1Net = { architecture: { ...arch }, training: baseTraining, trainingData: { text } };
  let phase1Last = null;
  const r1 = await trainNetwork(phase1Net, { onProgress: p => phase1Last = p });
  check('charLM-cont: phase1 produced state', !!r1.state);
  check('charLM-cont: phase1 returned architecture with vocabSize set',
        r1.architecture && r1.architecture.vocabSize > 0,
        'arch.vocabSize=' + r1.architecture?.vocabSize);
  const realVocab = r1.architecture.vocabSize;

  // Now simulate the bug condition: arch.vocabSize on disk was NEVER updated
  // (this is what older versions did). Continue training with the stale arch.
  const continueNet = {
    architecture: { ...arch, vocabSize: 0 },  // stale on-disk arch
    training: { ...baseTraining, epochs: 2 },
    trainingData: { text },
    state: r1.state,
    optimizerState: r1.optimizerState,
    tokenizer: r1.tokenizer
  };
  let firstStepLoss = null, contLast = null;
  const r2 = await trainNetwork(continueNet, { onProgress: p => {
    if (firstStepLoss === null) firstStepLoss = p.loss;
    contLast = p;
  } });
  check('charLM-cont: continue produced state', !!r2.state);
  // The phase-1 final loss is our reference. If we wiped the model, the first
  // continue step would jump back to ~log(vocabSize) ≈ 3-4 for this corpus.
  // We assert it stays close to where we left off.
  const phase1End = phase1Last.loss;
  check('charLM-cont: first continue step loss stays close to phase1 final (no model wipe)',
        firstStepLoss < phase1End + 1.0,
        `phase1End=${phase1End?.toFixed(3)} firstContinue=${firstStepLoss?.toFixed(3)} (gap > 1.0 means weights were wiped)`);
  // And the on-the-wire arch in r2 should now have the correct vocabSize too.
  check('charLM-cont: continue returned arch with corrected vocabSize',
        r2.architecture && r2.architecture.vocabSize === realVocab,
        'got=' + r2.architecture?.vocabSize + ' expected=' + realVocab);
}

async function testTokenizerRebuildOnNewChars() {
  console.log('\n== charLM tokenizer handles new chars on retrain ==');
  // Reproduces the bug: train on a small corpus, add a sample with chars
  // the original vocab never saw, then re-train. Both fromScratch and
  // continue should succeed — fromScratch by rebuilding the tokenizer,
  // continue by extending it.
  const arch = { kind: 'charLM', vocabSize: 0, embDim: 8, contextLen: 8, hidden: [16], activation: 'gelu', dropout: 0 };
  const tr  = { optimizer: 'adam', learningRate: 0.01, batchSize: 8, epochs: 2, seed: 42 };

  // Phase 1: train on lowercase-only text.
  const initial = { architecture: { ...arch }, training: tr,
    trainingData: { text: 'hello world. lowercase only here.\n'.repeat(20) } };
  const r1 = await trainNetwork(initial, {});
  check('phase1: tokenizer built', !!r1.tokenizer);
  const initialVocabSize = r1.tokenizer.chars.length;
  check('phase1: vocab is lowercase-only',
        !r1.tokenizer.chars.includes('E') && !r1.tokenizer.chars.includes('X'),
        'chars=' + JSON.stringify(r1.tokenizer.chars));

  // Phase 2a: simulate "Train from scratch" with new corpus that introduces
  // 'E' and 'X'. Old code would crash here with: char not in vocab: "E".
  const scratchNet = {
    architecture: { ...arch, vocabSize: initialVocabSize, isChat: false },
    training: tr,
    trainingData: { text: 'EXAMPLE text with NEW caps. hello world.\n'.repeat(20) },
    state: r1.state,
    tokenizer: r1.tokenizer
  };
  let scratchOk = false, scratchErr = null;
  try {
    const r2 = await trainNetwork(scratchNet, { fromScratch: true });
    scratchOk = !!r2.state && !!r2.tokenizer;
    check('fromScratch: vocab includes new "E"', r2.tokenizer.chars.includes('E'));
    check('fromScratch: vocab includes new "X"', r2.tokenizer.chars.includes('X'));
  } catch (e) { scratchErr = e.message; }
  check('fromScratch: training completes despite new chars in corpus', scratchOk, scratchErr);

  // Phase 2b: continue training with new chars in corpus. The vocab should
  // be extended (not reset), and training should not crash.
  const continueNet = {
    architecture: { ...arch, vocabSize: initialVocabSize, isChat: false },
    training: tr,
    trainingData: { text: 'EXAMPLE text with NEW caps. hello world.\n'.repeat(20) },
    state: r1.state,
    tokenizer: r1.tokenizer
  };
  let continueOk = false, continueErr = null;
  let logs = [];
  try {
    const r3 = await trainNetwork(continueNet, { log: (l) => logs.push(l) });
    continueOk = !!r3.state && !!r3.tokenizer;
    check('continue: vocab extended (size grew)', r3.tokenizer.chars.length > initialVocabSize,
          `was=${initialVocabSize} now=${r3.tokenizer.chars.length}`);
    // append-only: original chars keep their positions
    let stable = true;
    for (let i = 0; i < initialVocabSize; i++) {
      if (r3.tokenizer.chars[i] !== r1.tokenizer.chars[i]) { stable = false; break; }
    }
    check('continue: existing token IDs stable (append-only extension)', stable);
    check('continue: emitted vocab-expansion log',
          logs.some(l => /vocab expanded/i.test(l)),
          'logs=' + JSON.stringify(logs));
  } catch (e) { continueErr = e.message; }
  check('continue: training completes despite new chars in corpus', continueOk, continueErr);
}

async function testEncryption() {
  console.log('\n== storage + encryption (isolated) ==');
  const path = require('path');
  const fs = require('fs');
  const os = require('os');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ncstore-'));
  const { Storage } = require('../src/main/storage');
  const st = new Storage(dir);
  let net = st.createNetwork({ name: 'Enc Test' });
  net = st.saveTrainedState(net.id, { state: { kind: 'classifier', layers: [] } });
  check('initial state saved', !!net.state);
  const updated = st.updateNetwork(net.id, { encryptionIntent: 'enable', passphrase: 'secret123' });
  check('encrypted flag set', updated.encrypted === true);
  const locked = st.getNetwork(net.id);
  check('state hidden without passphrase', locked.stateLocked === true && !locked.state);
  const unlocked = st.getNetwork(net.id, 'secret123');
  check('decrypt succeeds', !unlocked.stateLocked && !!unlocked.state);
  const bad = st.getNetwork(net.id, 'wrong');
  check('wrong passphrase fails', bad.stateLocked === true);
  // cleanup
  fs.rmSync(dir, { recursive: true, force: true });
}

async function testParallelTrainerProducesValidModel() {
  console.log('\n== parallel trainer (worker_threads) ==');
  // Build a representative charLM corpus that all workers can chew through.
  const text = ('the quick brown fox jumps over the lazy dog. ' +
                'neural networks learn patterns from data. ' +
                'every weight is a tiny vote.\n').repeat(20);
  const net = {
    architecture: { kind: 'charLM', vocabSize: 0, embDim: 12, contextLen: 12, hidden: [32], activation: 'gelu', dropout: 0 },
    training: { optimizer: 'adam', learningRate: 0.005, batchSize: 16, epochs: 4, seed: 42, workers: 2 },
    trainingData: { text },
    state: null, tokenizer: null
  };
  const result = await trainNetwork(net);
  check('parallel: produced state', !!result.state, 'no state returned');
  check('parallel: produced tokenizer', !!result.tokenizer, 'no tokenizer returned');
  check('parallel: returned architecture', !!result.architecture && result.architecture.vocabSize > 0);
  check('parallel: metrics emitted', Array.isArray(result.metrics) && result.metrics.length === 4);
  // Loss should drop below the random-guess baseline (≈ ln(vocabSize)).
  const last = result.metrics[result.metrics.length - 1].loss;
  const baseline = Math.log(result.architecture.vocabSize);
  check('parallel: loss dropped below random baseline', last < baseline * 0.8,
        `last=${last.toFixed(3)} baseline=${baseline.toFixed(3)}`);

  // Inference round-trip — proves the saved state is valid.
  const trained = { ...net, state: result.state, tokenizer: result.tokenizer, architecture: result.architecture };
  const out = await infer(trained, { prompt: 'the ', maxTokens: 10 });
  check('parallel: inference produces text', typeof out.text === 'string' && out.text.length > 0);
}

(async () => {
  console.log('NeuralCabin engine tests');
  try {
    await testTensor();
    await testMLP();
    await testCharLM();
    await testDSL();
    await testChatFormat();
    await testChatTrainingEndToEnd();
    await testOptimizerStateRoundtrip();
    await testContinueTrainingPreservesState();
    await testInferencePaddingChoice();
    await testCharLMContinuePreservesWeights();
    await testTokenizerRebuildOnNewChars();
    await testMultiTurnChatFormat();
    await testMultiTurnInferenceEndToEnd();
    await testParallelTrainerProducesValidModel();
    await testEncryption();
  } catch (e) {
    console.error('Suite crashed:', e.stack || e);
    failed++;
  }
  console.log(`\n${passed} passed · ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})();
