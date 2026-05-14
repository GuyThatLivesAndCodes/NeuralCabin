export default function PluginsTab() {
  return (
    <div className="tab-content">
      <h2>🔌 Plugins</h2>
      <p style={{ color: '#7d6b5f', marginBottom: '20px' }}>
        Extend NeuralCabin with custom plugins for advanced features.
      </p>

      <div className="card">
        <h3>Coming Soon</h3>
        <p style={{ color: '#7d6b5f', marginBottom: '12px' }}>
          The plugin system will allow you to:
        </p>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>Create custom network architectures</li>
          <li>Implement new loss functions</li>
          <li>Add custom optimizers</li>
          <li>Integrate external data sources</li>
          <li>Build specialized UI components</li>
        </ul>
      </div>

      <div className="card">
        <h3>Plugin Types</h3>
        <div style={{ display: 'grid', gap: '8px' }}>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Custom Layers</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Define new neural network layer types
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Loss Functions</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Implement custom training objectives
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Data Loaders</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Add support for different data formats
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>UI Extensions</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Create custom tabs and visualizations
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Get Started with Plugins</h3>
        <p style={{ color: '#7d6b5f', marginBottom: '12px' }}>
          Check out the documentation for plugin development guidelines and examples.
        </p>
        <button style={{ width: '100%', padding: '10px' }}>📖 Plugin Development Guide</button>
      </div>
    </div>
  )
}
