export default function DocsTab() {
  return (
    <div className="tab-content">
      <h2>📖 Documentation</h2>

      <div className="card">
        <h3>Getting Started in 4 Steps</h3>
        <ol style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>
            <strong>Create a Network</strong> - Go to the <span style={{ color: '#d65a2a' }}>Networks</span> tab and click "Create New Network"
          </li>
          <li>
            <strong>Create a Dataset</strong> - In the <span style={{ color: '#d65a2a' }}>Training</span> tab, create an XOR dataset
          </li>
          <li>
            <strong>Start Training</strong> - Select your network and dataset, then click "Start Training"
          </li>
          <li>
            <strong>Monitor Progress</strong> - Watch the loss curve update in real-time as your network trains
          </li>
        </ol>
      </div>

      <div className="card">
        <h3>Tab Guide</h3>
        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ borderLeft: '4px solid #d65a2a', paddingLeft: '12px' }}>
            <strong style={{ color: '#d65a2a' }}>🧠 Networks</strong>
            <p style={{ fontSize: '14px', margin: '4px 0 0 0' }}>Create, view, and delete neural networks</p>
          </div>
          <div style={{ borderLeft: '4px solid #d65a2a', paddingLeft: '12px' }}>
            <strong style={{ color: '#d65a2a' }}>⚡ Training</strong>
            <p style={{ fontSize: '14px', margin: '4px 0 0 0' }}>Configure and monitor training sessions with live loss plotting</p>
          </div>
          <div style={{ borderLeft: '4px solid #d65a2a', paddingLeft: '12px' }}>
            <strong style={{ color: '#d65a2a' }}>📚 Corpus</strong>
            <p style={{ fontSize: '14px', margin: '4px 0 0 0' }}>Manage training datasets (coming soon)</p>
          </div>
          <div style={{ borderLeft: '4px solid #d65a2a', paddingLeft: '12px' }}>
            <strong style={{ color: '#d65a2a' }}>📝 Vocabulary</strong>
            <p style={{ fontSize: '14px', margin: '4px 0 0 0' }}>Configure vocabularies for sequence models (coming soon)</p>
          </div>
          <div style={{ borderLeft: '4px solid #d65a2a', paddingLeft: '12px' }}>
            <strong style={{ color: '#d65a2a' }}>🔮 Inference</strong>
            <p style={{ fontSize: '14px', margin: '4px 0 0 0' }}>Test trained models with custom inputs (coming soon)</p>
          </div>
          <div style={{ borderLeft: '4px solid #d65a2a', paddingLeft: '12px' }}>
            <strong style={{ color: '#d65a2a' }}>🔌 Plugins</strong>
            <p style={{ fontSize: '14px', margin: '4px 0 0 0' }}>Extend NeuralCabin with custom plugins (coming soon)</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Technology Stack</h3>
        <p style={{ color: '#7d6b5f', marginBottom: '12px' }}>
          NeuralCabin is built with modern, pure-Rust technology for maximum performance and safety.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <p style={{ fontWeight: '600', color: '#d65a2a', marginBottom: '4px' }}>Backend</p>
            <p style={{ fontSize: '13px' }}>Rust • Axum • Tokio</p>
          </div>
          <div>
            <p style={{ fontWeight: '600', color: '#d65a2a', marginBottom: '4px' }}>Frontend</p>
            <p style={{ fontSize: '13px' }}>React • TypeScript • Vite</p>
          </div>
          <div>
            <p style={{ fontWeight: '600', color: '#d65a2a', marginBottom: '4px' }}>Engine</p>
            <p style={{ fontSize: '13px' }}>Pure Rust • No External Math</p>
          </div>
          <div>
            <p style={{ fontWeight: '600', color: '#d65a2a', marginBottom: '4px' }}>Communication</p>
            <p style={{ fontSize: '13px' }}>REST API • WebSocket</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Current Features</h3>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>✅ Create and manage neural networks</li>
          <li>✅ Configure and run training sessions</li>
          <li>✅ Real-time loss plotting via WebSocket</li>
          <li>✅ Multiple optimizer support (Adam, SGD)</li>
          <li>✅ Configurable epochs and batch size</li>
        </ul>
      </div>

      <div className="card">
        <h3>Coming Soon</h3>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>Model persistence and loading</li>
          <li>Corpus/dataset file upload</li>
          <li>Vocabulary editor</li>
          <li>Inference/generation features</li>
          <li>Plugin system</li>
          <li>Training pause/resume</li>
        </ul>
      </div>
    </div>
  )
}
