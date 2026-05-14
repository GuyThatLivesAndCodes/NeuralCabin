export default function CorpusTab() {
  return (
    <div className="tab-content">
      <h2>📚 Corpus</h2>
      <p style={{ color: '#7d6b5f', marginBottom: '20px' }}>
        Manage your training datasets and corpora.
      </p>

      <div className="card">
        <h3>Coming Soon</h3>
        <p style={{ color: '#7d6b5f', marginBottom: '12px' }}>
          This feature is currently under development. Soon you'll be able to:
        </p>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>Upload custom datasets</li>
          <li>Preview data samples</li>
          <li>Split data into train/validation sets</li>
          <li>Visualize dataset statistics</li>
          <li>Create synthetic datasets</li>
        </ul>
      </div>

      <div className="card">
        <h3>Featured Datasets</h3>
        <div className="list-item">
          <div>
            <strong style={{ color: '#d65a2a' }}>XOR Dataset</strong>
            <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
              Classic binary classification dataset
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
