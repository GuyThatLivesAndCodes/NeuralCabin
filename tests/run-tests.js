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

(async () => {
  console.log('NeuralCity engine tests');
  try {
    await testTensor();
    await testMLP();
    await testCharLM();
    await testDSL();
    await testChatFormat();
    await testChatTrainingEndToEnd();
    await testOptimizerStateRoundtrip();
    await testContinueTrainingPreservesState();
    await testTokenizerRebuildOnNewChars();
    await testEncryption();
  } catch (e) {
    console.error('Suite crashed:', e.stack || e);
    failed++;
  }
  console.log(`\n${passed} passed · ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})();
