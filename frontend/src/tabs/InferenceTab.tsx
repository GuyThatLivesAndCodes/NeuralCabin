export default function InferenceTab() {
  return (
    <div className="tab-content">
      <h2>🔮 Inference</h2>
      <p style={{ color: '#7d6b5f', marginBottom: '20px' }}>
        Run inference with trained models and test predictions.
      </p>

      <div className="card">
        <h3>Coming Soon</h3>
        <p style={{ color: '#7d6b5f', marginBottom: '12px' }}>
          Once you've trained a model, you'll be able to:
        </p>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>Select a trained model</li>
          <li>Input custom data for prediction</li>
          <li>Visualize predictions and confidence scores</li>
          <li>Batch inference on multiple inputs</li>
          <li>Export prediction results</li>
        </ul>
      </div>

      <div className="card">
        <h3>Inference Modes</h3>
        <div style={{ display: 'grid', gap: '8px' }}>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Single Input</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Predict on a single input at a time
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Batch Inference</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Run predictions on multiple inputs efficiently
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Generation</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Generate sequences (for language models)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
