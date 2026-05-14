export default function DocsTab() {
  return (
    <div className="tab-content">
      <h2>Documentation</h2>
      <div className="card">
        <h3>Getting Started</h3>
        <ol>
          <li>Create a network in the <strong>Networks</strong> tab</li>
          <li>Create a dataset in the <strong>Training</strong> tab (or use existing one)</li>
          <li>Start training in the <strong>Training</strong> tab to see live loss updates</li>
          <li>Monitor training progress and loss curves in real-time</li>
        </ol>
      </div>

      <div className="card">
        <h3>Architecture</h3>
        <p>NeuralCabin is built with:</p>
        <ul>
          <li><strong>Backend:</strong> Rust with Axum web framework</li>
          <li><strong>Engine:</strong> Pure Rust neural network library</li>
          <li><strong>Frontend:</strong> React with TypeScript</li>
          <li><strong>Communication:</strong> REST API + WebSocket for real-time updates</li>
        </ul>
      </div>

      <div className="card">
        <h3>Current Limitations</h3>
        <p>This is an MVP demonstrating the architecture:</p>
        <ul>
          <li>Only XOR dataset for training</li>
          <li>Fixed network architecture (2-8-1 with tanh and sigmoid)</li>
          <li>No model persistence yet</li>
          <li>No inference/generation features yet</li>
        </ul>
      </div>
    </div>
  )
}
