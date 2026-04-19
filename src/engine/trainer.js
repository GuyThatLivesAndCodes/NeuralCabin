'use strict';

const T = require('./tensor');
const { buildFromSpec, restoreFromState, CharLM } = require('./model');
const { buildOptim } = require('./optim');
const { CharTokenizer } = require('./tokenizer');
const ChatFormat = require('./chat-format');

// Top-level entry: run one training session for a network config.
// network: the full stored network object.
// hooks: { onProgress({epoch, step, totalSteps, loss, lr, elapsedMs}), shouldStop() => bool }
// Returns: { state, metrics, tokenizer? }
async function trainNetwork(network, hooks = {}) {
  // Async wrapper that yields periodically by polling a generator.
  const gen = _trainCoreGen(network, hooks);
  let result;
  while (true) {
    const step = gen.next();
    if (step.done) { result = step.value; break; }
    await new Promise(r => setImmediate(r));
  }
  return result;
}

function trainNetworkSync(network, hooks = {}) {
  // Runs the generator to completion without yielding.
  const gen = _trainCoreGen(network, hooks);
  let step;
  while (!(step = gen.next()).done) {}
  return step.value;
}

function* _trainCoreGen(network, hooks) {
  const onProgress = hooks.onProgress || (() => {});
  const shouldStop = hooks.shouldStop || (() => false);
  const log = hooks.log || (() => {});
  const fromScratch = !!hooks.fromScratch;

  const seed = network.training?.seed ?? 42;
  const rng = T.rngFromSeed(seed);

  // Build or restore model.
  // - If fromScratch is set, ignore any saved state and rebuild fresh.
  // - Otherwise, load saved weights (continuation training).
  let model;
  let resumed = false;
  if (network.state && !fromScratch) {
    model = restoreFromState(network.state, network.architecture, rng);
    resumed = true;
  } else {
    model = buildFromSpec(network.architecture, rng);
  }

  // Build optimizer. Restore momentum/variance buffers if continuing — this
  // prevents the loss spike you'd otherwise see on the first few steps after
  // resume, because Adam would restart its running statistics from zero.
  const optName = network.training?.optimizer || 'adam';
  const lr = network.training?.learningRate ?? 1e-3;
  const optim = buildOptim(optName, model.params, { lr });
  let restoredOptim = false;
  if (resumed && network.optimizerState) {
    restoredOptim = optim.loadFromJSON(network.optimizerState);
    if (restoredOptim) log(`continuing from saved weights (optimizer state restored, ${optim.t || 0} prior steps)`);
    else log('continuing from saved weights (optimizer state mismatched — starting optimizer fresh)');
  } else if (resumed) {
    log('continuing from saved weights (no optimizer state — starting optimizer fresh)');
  } else if (fromScratch && network.state) {
    log('training from scratch (existing weights discarded)');
  }

  // Prepare data based on kind
  const arch = network.architecture;
  const data = network.trainingData || {};
  const batchSize = network.training?.batchSize || 32;
  const epochs = network.training?.epochs || 20;

  const start = Date.now();
  const metrics = [];

  if (arch.kind === 'classifier' || arch.kind === 'mlp' || arch.kind === 'regressor') {
    const samples = data.samples || [];
    if (samples.length === 0) throw new Error('No training samples provided');
    const isRegression = arch.kind === 'regressor';

    // Build X and y arrays
    const N = samples.length;
    const X = new Float32Array(N * arch.inputDim);
    let Y; // either Float32Array (regression) or int[] (classification)
    if (isRegression) Y = new Float32Array(N * arch.outputDim);
    else Y = new Array(N);

    for (let i = 0; i < N; i++) {
      const sample = samples[i];
      if (!Array.isArray(sample.input) || sample.input.length !== arch.inputDim) {
        throw new Error(`sample ${i} input length ${sample.input?.length} != inputDim ${arch.inputDim}`);
      }
      for (let j = 0; j < arch.inputDim; j++) X[i * arch.inputDim + j] = sample.input[j];
      if (isRegression) {
        if (!Array.isArray(sample.output) || sample.output.length !== arch.outputDim) {
          throw new Error(`sample ${i} output length invalid`);
        }
        for (let j = 0; j < arch.outputDim; j++) Y[i * arch.outputDim + j] = sample.output[j];
      } else {
        const label = typeof sample.label === 'number' ? sample.label : sample.output;
        if (typeof label !== 'number') throw new Error(`sample ${i} missing numeric label`);
        Y[i] = label;
      }
    }

    // Index shuffle buffer
    const idx = new Int32Array(N);
    for (let i = 0; i < N; i++) idx[i] = i;

    const stepsPerEpoch = Math.max(1, Math.floor(N / batchSize));
    const totalSteps = epochs * stepsPerEpoch;
    let globalStep = 0;

    for (let ep = 0; ep < epochs; ep++) {
      // shuffle
      for (let i = N - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }
      let epLoss = 0;
      for (let s = 0; s < stepsPerEpoch; s++) {
        if (shouldStop()) return { state: model.toJSON(), optimizerState: optim.toJSON(), metrics, stopped: true };
        const xbData = new Float32Array(batchSize * arch.inputDim);
        const ybLabels = isRegression ? new Float32Array(batchSize * arch.outputDim) : new Array(batchSize);
        for (let b = 0; b < batchSize; b++) {
          const sIdx = idx[(s * batchSize + b) % N];
          for (let j = 0; j < arch.inputDim; j++) xbData[b * arch.inputDim + j] = X[sIdx * arch.inputDim + j];
          if (isRegression) {
            for (let j = 0; j < arch.outputDim; j++) ybLabels[b * arch.outputDim + j] = Y[sIdx * arch.outputDim + j];
          } else {
            ybLabels[b] = Y[sIdx];
          }
        }
        const xb = new T.Tensor([batchSize, arch.inputDim], xbData, false);
        optim.zeroGrad();
        const logits = model.forward(xb, { training: true, rng });
        let loss;
        if (isRegression) {
          const yb = new T.Tensor([batchSize, arch.outputDim], ybLabels, false);
          loss = T.mseLoss(logits, yb);
        } else {
          loss = T.softmaxCrossEntropy(logits, ybLabels);
        }
        loss.backward();
        optim.step();
        epLoss += loss.data[0];
        globalStep++;

        if (globalStep % Math.max(1, Math.floor(totalSteps / 200)) === 0 || globalStep === totalSteps) {
          onProgress({
            epoch: ep + 1, totalEpochs: epochs,
            step: globalStep, totalSteps,
            loss: loss.data[0],
            elapsedMs: Date.now() - start
          });
          yield;
        }
      }
      metrics.push({ epoch: ep + 1, loss: epLoss / stepsPerEpoch });
      log(`epoch ${ep + 1}/${epochs}  loss=${(epLoss / stepsPerEpoch).toFixed(6)}`);
    }

    return { state: model.toJSON(), optimizerState: optim.toJSON(), metrics };
  }

  if (arch.kind === 'charLM') {
    // text-based: data.text OR data.samples:[{text}] OR chat-shaped samples
    const corpus = ChatFormat.buildCorpus(data);
    const text = corpus.text;
    if (text.length < arch.contextLen + 2) {
      throw new Error(corpus.isChat
        ? `Chat corpus too short for contextLen=${arch.contextLen}. Add more sample pairs or shorten contextLen.`
        : `Training text too short for contextLen=${arch.contextLen}.`);
    }
    // Mark on the architecture so inference knows to use chat formatting.
    arch.isChat = corpus.isChat;

    // Build/keep tokenizer.
    //
    // Three cases:
    //   1. fromScratch: rebuild vocab from the *current* corpus. The saved
    //      tokenizer might be stale (e.g. user added a sample with new chars
    //      and clicked "Train from scratch") — reusing it here would crash
    //      with "char not in vocab" when we encode the corpus below.
    //   2. continue with saved tokenizer that already covers the corpus:
    //      keep the saved tokenizer as-is so token IDs stay aligned with
    //      the embedding/output rows the trained weights expect.
    //   3. continue but the corpus contains new chars: extend the vocab
    //      (append-only so existing IDs don't shift), then rebuild the model
    //      from scratch — the embedding and output-projection dims have to
    //      grow to fit the new vocab, and there's no clean way to splice
    //      new rows into the saved weights. We log this so the user knows
    //      why the loss restarts high in this specific case.
    let tokenizer;
    if (network.tokenizer && !fromScratch) {
      tokenizer = CharTokenizer.fromJSON(network.tokenizer);
      const known = new Set(tokenizer.chars);
      const novel = [];
      for (const ch of text) if (!known.has(ch)) { known.add(ch); novel.push(ch); }
      if (novel.length > 0) {
        // Append-only extension keeps existing token IDs stable.
        const extended = tokenizer.chars.concat(novel);
        tokenizer = new CharTokenizer(extended);
        const preview = novel.slice(0, 8).map(c => JSON.stringify(c)).join(', ');
        log(`vocab expanded by ${novel.length} new char(s) [${preview}${novel.length > 8 ? ', …' : ''}] — rebuilding model from scratch to fit new vocab size`);
        arch.vocabSize = tokenizer.vocabSize;
        model = buildFromSpec(arch, rng);
        const newOpt = buildOptim(optName, model.params, { lr });
        Object.assign(optim, newOpt);
        resumed = false;
      }
    } else {
      tokenizer = CharTokenizer.fromCorpus(text);
    }
    if (tokenizer.vocabSize !== arch.vocabSize) {
      arch.vocabSize = tokenizer.vocabSize;
      model = buildFromSpec(arch, rng);
      const newOpt = buildOptim(optName, model.params, { lr });
      Object.assign(optim, newOpt);
    }

    const ids = tokenizer.encode(text);
    const L = arch.contextLen;
    const N = ids.length - L;
    if (N <= 0) throw new Error('not enough tokens');

    const stepsPerEpoch = Math.max(1, Math.floor(N / batchSize));
    const totalSteps = epochs * stepsPerEpoch;
    let globalStep = 0;

    for (let ep = 0; ep < epochs; ep++) {
      let epLoss = 0;
      for (let s = 0; s < stepsPerEpoch; s++) {
        if (shouldStop()) return { state: model.toJSON(), optimizerState: optim.toJSON(), tokenizer: tokenizer.toJSON(), metrics, stopped: true };
        const idsBatch = [];
        const labels = [];
        for (let b = 0; b < batchSize; b++) {
          const start = Math.floor(rng() * N);
          const ctx = ids.slice(start, start + L);
          const nxt = ids[start + L];
          idsBatch.push(ctx);
          labels.push(nxt);
        }
        optim.zeroGrad();
        const logits = model.forward(idsBatch, { training: true, rng });
        const loss = T.softmaxCrossEntropy(logits, labels);
        loss.backward();
        optim.step();
        epLoss += loss.data[0];
        globalStep++;
        if (globalStep % Math.max(1, Math.floor(totalSteps / 200)) === 0 || globalStep === totalSteps) {
          onProgress({
            epoch: ep + 1, totalEpochs: epochs,
            step: globalStep, totalSteps,
            loss: loss.data[0],
            elapsedMs: Date.now() - start
          });
          yield;
        }
      }
      metrics.push({ epoch: ep + 1, loss: epLoss / stepsPerEpoch });
      log(`epoch ${ep + 1}/${epochs}  loss=${(epLoss / stepsPerEpoch).toFixed(6)}`);
    }
    return { state: model.toJSON(), optimizerState: optim.toJSON(), tokenizer: tokenizer.toJSON(), metrics };
  }

  throw new Error('unknown arch kind ' + arch.kind);
}

// Inference entry. Returns a JSON-able result.
function infer(network, input) {
  const rng = T.rngFromSeed(network.training?.seed ?? 42);
  if (!network.state) throw new Error('Network has no trained state');
  const model = restoreFromState(network.state, network.architecture, rng);
  const arch = network.architecture;

  if (arch.kind === 'classifier' || arch.kind === 'mlp') {
    const vec = input.input || input;
    if (!Array.isArray(vec) || vec.length !== arch.inputDim) throw new Error('input must be array of length ' + arch.inputDim);
    const x = new T.Tensor([1, arch.inputDim], new Float32Array(vec), false);
    const logits = model.forward(x, { training: false });
    const probs = T.softmax(logits);
    const arr = Array.from(probs.data);
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return {
      kind: 'classification',
      probs: arr,
      predictedClass: best,
      label: (arch.classes || [])[best] || String(best)
    };
  }

  if (arch.kind === 'regressor') {
    const vec = input.input || input;
    const x = new T.Tensor([1, arch.inputDim], new Float32Array(vec), false);
    const out = model.forward(x, { training: false });
    return { kind: 'regression', output: Array.from(out.data) };
  }

  if (arch.kind === 'charLM') {
    if (!network.tokenizer) throw new Error('tokenizer missing');
    const tokenizer = CharTokenizer.fromJSON(network.tokenizer);
    const userPrompt = String(input.prompt ?? input ?? '');
    const maxNew = input.maxTokens ?? 120;
    const temperature = input.temperature ?? 1.0;
    const topK = input.topK ?? 0;

    // If the model was trained on chat data, wrap the prompt in role tags
    // and stop generation at the assistant <|end|> tag.
    const isChat = !!arch.isChat || input.chat === true;
    const promptText = isChat
      ? ChatFormat.wrapPromptForChat(userPrompt, input.system || '')
      : userPrompt;

    // Encode (silently dropping unseen chars).
    const L = arch.contextLen;
    let ids = tokenizer.encodeSafe(promptText);
    if (ids.length === 0) ids = [0];
    while (ids.length < L) ids = [ids[0], ...ids];

    // Pre-encode the END tag so we can detect it byte-for-byte in the output.
    const endTag = ChatFormat.TAGS.END;
    const endIds = tokenizer.encodeSafe(endTag);
    const stopOnEnd = isChat && endIds.length > 0;

    const out = [];
    for (let step = 0; step < maxNew; step++) {
      const ctx = ids.slice(ids.length - L);
      const logits = model.forward([ctx], { training: false });
      const row = Array.from(logits.data);
      for (let i = 0; i < row.length; i++) row[i] /= Math.max(temperature, 1e-6);
      let indices = row.map((v, i) => i);
      if (topK > 0 && topK < row.length) {
        indices.sort((a, b) => row[b] - row[a]);
        indices = indices.slice(0, topK);
      }
      const maxv = Math.max(...indices.map(i => row[i]));
      let sum = 0;
      const ex = indices.map(i => { const e = Math.exp(row[i] - maxv); sum += e; return e; });
      const probs = ex.map(e => e / sum);
      let r = Math.random();
      let pick = indices[indices.length - 1];
      for (let i = 0; i < indices.length; i++) {
        r -= probs[i];
        if (r <= 0) { pick = indices[i]; break; }
      }
      ids.push(pick);
      out.push(pick);
      // Stop if we just generated the END tag.
      if (stopOnEnd && out.length >= endIds.length) {
        let match = true;
        for (let k = 0; k < endIds.length; k++) {
          if (out[out.length - endIds.length + k] !== endIds[k]) { match = false; break; }
        }
        if (match) {
          out.length -= endIds.length; // strip the tag from output
          break;
        }
      }
    }
    const generated = tokenizer.decode(out);
    if (isChat) {
      // Final safety: strip any leftover tag content the model produced.
      const reply = ChatFormat.extractAssistantReply(generated);
      return { kind: 'generation', text: reply, raw: generated, tokens: out, chat: true };
    }
    return { kind: 'generation', text: generated, tokens: out };
  }

  throw new Error('unknown arch kind ' + arch.kind);
}

module.exports = { trainNetwork, trainNetworkSync, infer };
