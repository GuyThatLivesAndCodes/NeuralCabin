window.NC_DOCS = [
  {
    id: 'welcome',
    title: 'Welcome',
    body: `
<h1>Welcome to NeuralCity</h1>
<p>NeuralCity lets you build neural networks from the ground up — no NumPy, no TensorFlow, no PyTorch. Every tensor operation, every gradient, every optimizer step in this app runs on plain JavaScript code that you can read and understand.</p>
<p>Pick a topic on the left to learn how the engine works, how to design networks, and how to use the built-in scripting language to experiment.</p>
<h2>Quick start</h2>
<ol>
  <li>Click <code>+ New</code> on the left and pick a template — <b>XOR Classifier</b> trains in a second.</li>
  <li>Open the <b>Train</b> tab and press <b>Start Training</b>.</li>
  <li>Switch to <b>Inference</b> to run predictions.</li>
  <li>Switch to <b>API</b> to expose it over HTTP on your local network.</li>
</ol>
`
  },
  {
    id: 'how-it-works',
    title: 'How networks learn',
    body: `
<h1>How a neural network actually learns</h1>
<p>A neural network is a chain of mathematical functions with <b>tunable knobs</b> (weights). Training is the process of repeatedly measuring how wrong the network is, then nudging every knob in the direction that makes it less wrong. That's it.</p>

<h2>1. Forward pass</h2>
<p>The input vector <code>x</code> flows through each layer. A <b>Linear</b> layer computes <code>y = x·W + b</code>. An <b>Activation</b> like <code>ReLU</code> squashes negatives to zero. Stack enough of these and the network can represent nearly any function.</p>

<h2>2. Loss</h2>
<p>We compare the network's prediction to the target and compute a single number measuring the error. For classification we use <b>softmax cross-entropy</b>: we turn logits into probabilities and then measure the negative log-probability of the correct class. For regression we use <b>mean squared error</b>.</p>

<h2>3. Backward pass (autograd)</h2>
<p>This is where the engine earns its keep. Every operation saved its parents when it ran forward. Starting from the loss (a scalar) we walk the graph in reverse, applying the chain rule to compute <code>∂loss/∂W</code> and <code>∂loss/∂b</code> for every parameter. Each operator knows its own local derivative — matmul, relu, softmax, and so on.</p>

<h2>4. Optimizer step</h2>
<p>Now every parameter has a gradient saying "increase me and the loss goes up." So we subtract a scaled version: <code>W ← W − lr · ∂loss/∂W</code>. <b>Adam</b> adds running averages of past gradients so training is smoother than plain SGD.</p>

<h2>5. Repeat</h2>
<p>Do that for thousands of mini-batches and the loss drifts down. The network has learned.</p>
`
  },
  {
    id: 'layers',
    title: 'Layers reference',
    body: `
<h1>Layers</h1>
<h2>Linear</h2>
<p><code>y = x·W + b</code>. Shape <code>[B, in] → [B, out]</code>. The workhorse.</p>
<h2>Activation</h2>
<table>
<tr><th>Name</th><th>Formula</th><th>When</th></tr>
<tr><td>relu</td><td>max(0, x)</td><td>Default for hidden layers.</td></tr>
<tr><td>leakyRelu</td><td>max(αx, x)</td><td>When ReLU kills too many neurons.</td></tr>
<tr><td>tanh</td><td>(e^x − e^−x)/(e^x + e^−x)</td><td>Smooth, bounded. Good for small regressors.</td></tr>
<tr><td>sigmoid</td><td>1 / (1 + e^−x)</td><td>Probabilities, binary outputs.</td></tr>
<tr><td>gelu</td><td>0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³)))</td><td>Modern default for language models.</td></tr>
<tr><td>softmax</td><td>e^xᵢ / Σe^xⱼ</td><td>Converts logits into a probability distribution.</td></tr>
</table>
<h2>Dropout</h2>
<p>During training, randomly zero <code>p</code> of the activations and scale the rest by <code>1/(1-p)</code>. Prevents co-adaptation and overfitting. Disabled at inference time.</p>
<h2>Embedding</h2>
<p>Look up a vector of length <code>dim</code> for each integer id. Used as the first layer of a language model to turn token ids into dense vectors.</p>
`
  },
  {
    id: 'data',
    title: 'Training data',
    body: `
<h1>Training data formats</h1>
<p>Each network kind expects a different shape in its <b>Training Data</b> tab.</p>
<h2>Classifier</h2>
<pre><code>[
  { "input": [0, 0], "label": 0 },
  { "input": [0, 1], "label": 1 },
  ...
]</code></pre>
<h2>Regressor</h2>
<pre><code>[
  { "input": [x1, x2], "output": [y1, y2] },
  ...
]</code></pre>
<h2>Character LM (free text)</h2>
<p>Either <code>{ "text": "whatever raw text..." }</code> or an array of <code>{ "text": "..." }</code> samples.</p>
<p>The LM builds a character-level vocabulary from your corpus. More text = a model with richer context.</p>

<h2>Chat Assistant (user → assistant pairs)</h2>
<p>Train a charLM on dialogue. Use <b>any</b> of these JSON shapes inside <code>samples</code>:</p>
<pre><code>{
  "samples": [
    { "user": "Hello, I need help", "assistant": "Of course, I'm here, what's up?" },
    { "user": "What can you do?",   "assistant": "I can chat about anything you teach me." }
  ]
}</code></pre>
<p>Multi-turn:</p>
<pre><code>{ "samples": [
  { "messages": [
      { "role": "system",    "content": "You are concise." },
      { "role": "user",      "content": "Hi" },
      { "role": "assistant", "content": "Hello!" }
  ] }
] }</code></pre>
<p>Or alternating turns:</p>
<pre><code>{ "samples": [
  { "conversation": [
      { "user": "Hi" },
      { "assistant": "Hey!" },
      { "user": "Tell me more" },
      { "assistant": "Sure — about what?" }
  ] }
] }</code></pre>
<p>Behind the scenes, NeuralCity flattens every conversation into a single text stream with role tags it learns to predict:</p>
<pre><code>&lt;|user|&gt;Hello, I need help&lt;|end|&gt;&lt;|assistant|&gt;Of course, I'm here, what's up?&lt;|end|&gt;</code></pre>
<p>At inference time the app wraps your message in the same tags and stops generation when the model emits <code>&lt;|end|&gt;</code>, so you get back just the assistant's reply.</p>
<p>Tip: bump <b>contextLen</b> (under Architecture) so a full user+assistant turn fits in the window. <code>48</code>–<code>96</code> is a good range for short chats; longer if your replies are paragraphs.</p>
`
  },
  {
    id: 'script',
    title: 'NeuralScript language',
    body: `
<h1>NeuralScript</h1>
<p>A tiny language built into NeuralCity for scripting experiments. No semicolons, no curly braces — uses <code>do</code>/<code>end</code>.</p>

<h2>Variables & expressions</h2>
<pre><code>let x = 10
let name = "hello"
set x = x + 1
print x</code></pre>

<h2>Control flow</h2>
<pre><code>if x &gt; 0 do
  print "positive"
else do
  print "non-positive"
end

while x &gt; 0 do
  set x = x - 1
end

for i = 0 to 9 do
  print i
end</code></pre>

<h2>Functions</h2>
<pre><code>fn square(n) do
  return n * n
end
print square(6)</code></pre>

<h2>Neural API</h2>
<p>Exposed standard library:</p>
<table>
<tr><th>Function</th><th>Purpose</th></tr>
<tr><td><code>build(spec)</code></td><td>Construct a model from an architecture spec.</td></tr>
<tr><td><code>await(train(spec, data, opts))</code></td><td>Train a model inline. Returns <code>{ state, metrics }</code>.</td></tr>
<tr><td><code>predict(network, input)</code></td><td>Run inference on a full network object.</td></tr>
<tr><td><code>thisNet()</code></td><td>Returns the currently selected network.</td></tr>
<tr><td><code>range</code>, <code>len</code>, <code>push</code>, <code>str</code>, <code>num</code>, <code>keys</code>, <code>values</code></td><td>Collection helpers.</td></tr>
<tr><td><code>abs</code>, <code>min</code>, <code>max</code>, <code>sqrt</code>, <code>exp</code>, <code>log</code>, <code>sin</code>, <code>cos</code>, <code>floor</code>, <code>ceil</code>, <code>round</code>, <code>random</code></td><td>Math helpers.</td></tr>
</table>
<h2>Example</h2>
<pre><code>let spec = {
  kind: "classifier",
  inputDim: 2, outputDim: 2,
  hidden: [8], activation: "relu",
  classes: ["false", "true"]
}
let data = { samples: [
  { input: [0,0], label: 0 },
  { input: [0,1], label: 1 },
  { input: [1,0], label: 1 },
  { input: [1,1], label: 0 }
] }
let opts = { optimizer: "adam", learningRate: 0.05, batchSize: 4, epochs: 200, seed: 42 }
let result = await(train(spec, data, opts))
print "done training"
print result.metrics[len(result.metrics) - 1]</code></pre>
`
  },
  {
    id: 'api',
    title: 'HTTP API',
    body: `
<h1>Serving models over HTTP</h1>
<p>Every trained network can be served on its own local port. Go to the <b>API</b> tab, select a network, pick a port (or leave <code>0</code> for auto), and click <b>Start</b>.</p>

<h2>Endpoints</h2>
<table>
<tr><th>Route</th><th>Method</th><th>Description</th></tr>
<tr><td><code>/</code> or <code>/info</code></td><td>GET</td><td>Returns network metadata and expected input shape.</td></tr>
<tr><td><code>/predict</code></td><td>POST</td><td>JSON body matching the input spec. Returns the prediction.</td></tr>
</table>

<h2>Examples</h2>
<p><b>Classifier / Regressor:</b></p>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input":[0,1]}'</code></pre>
<p><b>Character LM:</b></p>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"the ","maxTokens":80,"temperature":1.0}'</code></pre>
<p>Share your local IP with other devices on the same network to let them call your model. Your IP is shown in the status bar.</p>
`
  },
  {
    id: 'encryption',
    title: 'Encryption',
    body: `
<h1>Encrypting your network</h1>
<p>Each network can be encrypted at rest with a passphrase. Weights and tokenizer are serialized, wrapped with AES-256-GCM, and a scrypt-derived key protects the bundle. The app never stores your passphrase — lose it and the weights are unrecoverable (that's the point).</p>
<p>In the <b>Editor</b> tab, flip the <b>Encryption</b> toggle, enter a passphrase, and save. To train or infer on an encrypted network you'll be prompted to decrypt it first.</p>
<h2>Why</h2>
<p>Models you train on personal or client data are sensitive. Keeping them encrypted means a stolen laptop doesn't mean stolen weights.</p>
`
  },
  {
    id: 'philosophy',
    title: 'Why from scratch?',
    body: `
<h1>No frameworks. No fluff.</h1>
<p>Popular frameworks are wonderful but they hide the actual math behind tens of thousands of lines of abstraction and specialized kernels. For learning, for small and medium models, and for <i>truly understanding</i> what your network is doing, that abstraction is noise.</p>
<p>NeuralCity's engine is about <b>2,000 lines of plain JavaScript</b>. You can open <code>src/engine/tensor.js</code>, read the <code>matmul</code> function, and verify every multiplication. You can modify a gradient, add a new activation, try your own optimizer. The tools are yours.</p>
<h2>What we give up</h2>
<ul>
  <li><b>GPU acceleration.</b> Everything runs on CPU. Great for models up to a few million parameters; slow for huge ones.</li>
  <li><b>Distributed training.</b> One machine, one process. That's intentional.</li>
  <li><b>Bleeding-edge layers.</b> No Transformers out of the box. But the pieces are there and the DSL lets you compose them.</li>
</ul>
<h2>What we keep</h2>
<ul>
  <li><b>Readable source.</b> Every operator fits on one screen.</li>
  <li><b>Reproducible runs.</b> Seeded randomness across init, shuffling, sampling.</li>
  <li><b>Own your data.</b> Nothing leaves your machine unless you explicitly serve it over the API.</li>
</ul>
`
  }
];
