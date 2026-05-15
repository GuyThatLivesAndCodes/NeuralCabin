import { Network } from '../api'

export default function DocsTab({ networks: list }: { networks: Network[] }) {
  const ff = list.filter(n => n.kind === 'feedforward')
  const nt = list.filter(n => n.kind === 'next_token')
  const trained = list.filter(n => n.trained).length

  return (
    <div className="tab-content">
      <h2>Documentation</h2>
      <p className="muted">
        Honest reference for what NeuralCabin does and how its pieces fit
        together. The numbers below are live — they reflect the current state
        of your workspace.
      </p>

      <div className="card">
        <h3>Workspace status</h3>
        <table>
          <tbody>
            <tr><th>Total networks</th><td>{list.length}</td></tr>
            <tr><th>Feed-forward</th><td>{ff.length}</td></tr>
            <tr><th>Next-token</th><td>{nt.length}</td></tr>
            <tr><th>Trained at least once</th><td>{trained}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>What this app actually does</h3>
        <p>
          NeuralCabin trains small neural networks entirely in Rust, with no
          Python, no PyTorch, no NumPy, and no remote services. The engine
          implements:
        </p>
        <ul style={{ paddingLeft: 20, lineHeight: 1.7 }}>
          <li><code>Linear</code> (fully-connected) layers and five activations
              (<code>identity</code>, <code>relu</code>, <code>sigmoid</code>, <code>tanh</code>, <code>softmax</code>).</li>
          <li>A reverse-mode autograd tape with hand-written backward passes.</li>
          <li>Two losses: <code>MeanSquaredError</code> and softmax + <code>CrossEntropy</code>.</li>
          <li>Two optimizers: <code>Adam</code> and <code>SGD</code> (with optional momentum).</li>
          <li>Char-level and word-level tokenizers for text networks.</li>
        </ul>
        <p className="mt-2 muted">
          <strong>What this app is NOT:</strong> there is no transformer,
          no self-attention, no convolution, no RNN. Sequence modeling is a
          fully-connected MLP that consumes a sliding window of one-hot tokens
          and predicts the next token. It works for tiny corpora — don’t expect
          GPT-quality output.
        </p>
      </div>

      <div className="card">
        <h3>Workflow</h3>
        <ol style={{ paddingLeft: 20, lineHeight: 1.8 }}>
          <li><strong>Networks tab</strong>: design the architecture. Pick
              feed-forward (numeric in/out) or next-token (text). The hidden-layer
              spec field accepts a comma-separated list of dims and activations,
              e.g. <code>64,relu,32,relu</code>.</li>
          <li><strong>Corpus tab</strong>: attach training data. The form changes
              with the network kind:
              <ul style={{ paddingLeft: 20, marginTop: 6 }}>
                <li>Feed-forward: paste or upload CSV — each row is{' '}
                    <code>in_dim + out_dim</code> comma-separated numbers.</li>
                <li>Next-token, <em>pretraining</em>: bulk-upload one or more
                    <code>.txt</code> files. The vocabulary is built automatically.</li>
                <li>Next-token, <em>fine-tuning</em>: provide input/output pairs
                    in the editor or import a JSON file.</li>
              </ul></li>
          <li><strong>Vocabulary tab</strong>: inspect the vocabulary that the
              corpus produced for next-token networks. Export to JSON.</li>
          <li><strong>Training tab</strong>: pick epochs, batch size, optimizer,
              learning rate, and (for fine-tuning) whether to mask user tokens.
              Loss curves stream live as the model trains.</li>
          <li><strong>Inference tab</strong>: feed inputs (or a prompt) to a
              trained network. For next-token networks you also get per-token
              probabilities for transparency.</li>
        </ol>
      </div>

      <div className="card">
        <h3>Network types</h3>
        <div className="grid-2">
          <div>
            <h4>Feed-forward</h4>
            <p>Numeric vectors in, numeric vectors out. Use for regression
              (continuous targets) or classification (one-hot targets with a
              softmax / cross-entropy combo).</p>
            <p className="muted small mt-1">
              Examples: XOR, polynomial regression, small tabular classifiers.
            </p>
          </div>
          <div>
            <h4>Next-token prediction</h4>
            <p>Text in, text out — modelled as predicting the next token from a
              fixed-size window of preceding tokens. The window is one-hot
              encoded then fed through dense layers to <code>vocab_size</code> logits.</p>
            <p className="muted small mt-1">
              Two stages: <em>pretraining</em> on free-form text, then
              optional <em>fine-tuning</em> on input/output pairs.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Fine-tuning format</h3>
        <p>
          The fine-tuning JSON format used by the Corpus tab is a flat array of
          objects, each with an <code>input</code> and <code>output</code> string:
        </p>
        <pre style={{
          background: 'var(--bg-input)', padding: 12, borderRadius: 'var(--radius)',
          border: '1px solid var(--border)', overflow: 'auto', fontSize: 12,
        }}>{`[
  { "input": "what is 2 + 2?",   "output": "4" },
  { "input": "what is 3 + 5?",   "output": "8" },
  { "input": "is the sky blue?", "output": "yes" }
]`}</pre>
        <p className="mt-1">
          Internally each pair is encoded as{' '}
          <code>&lt;bos&gt; input &lt;eos&gt; output &lt;eos&gt;</code> and one
          training example is emitted per token transition. With <em>mask user
          tokens</em> on (the default), the model is only scored on producing
          the assistant output.
        </p>
      </div>

      <div className="card">
        <h3>Reserved vocabulary tokens</h3>
        <table>
          <thead><tr><th>ID</th><th>Token</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td><code>0</code></td><td><code>&lt;pad&gt;</code></td><td>Left-padding for short contexts.</td></tr>
            <tr><td><code>1</code></td><td><code>&lt;unk&gt;</code></td><td>Anything not in the trained vocab.</td></tr>
            <tr><td><code>2</code></td><td><code>&lt;bos&gt;</code></td><td>Beginning of sequence (fine-tune).</td></tr>
            <tr><td><code>3</code></td><td><code>&lt;eos&gt;</code></td><td>End of sequence; halts generation when sampled.</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Hidden-layer mini-language</h3>
        <p>
          The hidden-layer field on the Networks form takes a comma-separated
          list. Numbers become Linear layers (with the given output dim);
          words become activations.
        </p>
        <table>
          <thead><tr><th>Spec</th><th>Resulting layers</th></tr></thead>
          <tbody>
            <tr><td><code>8,tanh</code></td><td>Linear → 8 · tanh</td></tr>
            <tr><td><code>64,relu,32,relu</code></td><td>Linear → 64 · relu · Linear → 32 · relu</td></tr>
            <tr><td><code>(empty)</code></td><td>Direct projection from input to output (no hidden layer)</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>How training actually runs</h3>
        <ol style={{ paddingLeft: 20, lineHeight: 1.7 }}>
          <li>The frontend sends a <code>StartTrainingRequest</code> over Tauri IPC.</li>
          <li>The backend builds an <code>(X, Y)</code> tensor pair from the
              attached corpus — sliding windows for next-token networks, raw
              rows for feed-forward.</li>
          <li>An <code>Optimizer</code> is constructed sized to the model’s
              actual parameter shapes.</li>
          <li>For each epoch the example indices are shuffled (deterministic
              from <code>seed</code>) and processed in batches.</li>
          <li>Each batch runs a real autograd pass: the loss is built into a
              fresh tape, gradients flow backward, and the optimizer updates
              weights in place.</li>
          <li>After every epoch, a <code>training_update</code> event is emitted
              with the mean batch loss.</li>
          <li>When the run finishes, the network is marked as <code>trained</code>
              and inference becomes meaningful.</li>
        </ol>
      </div>
    </div>
  )
}
