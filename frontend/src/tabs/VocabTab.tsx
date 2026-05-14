export default function VocabTab() {
  return (
    <div className="tab-content">
      <h2>📝 Vocabulary</h2>
      <p style={{ color: '#7d6b5f', marginBottom: '20px' }}>
        Configure and manage vocabularies for sequence models and language tasks.
      </p>

      <div className="card">
        <h3>Coming Soon</h3>
        <p style={{ color: '#7d6b5f', marginBottom: '12px' }}>
          Vocabulary management tools will allow you to:
        </p>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>Create vocabularies from text corpora</li>
          <li>View and edit vocabulary entries</li>
          <li>Configure tokenization strategies</li>
          <li>Set special tokens (pad, unk, eos, etc.)</li>
          <li>Preview tokenization results</li>
        </ul>
      </div>

      <div className="card">
        <h3>Vocabulary Types</h3>
        <div style={{ display: 'grid', gap: '8px' }}>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Character-level</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                One vocabulary entry per character
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Byte Pair Encoding</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                Subword tokenization with merging
              </p>
            </div>
          </div>
          <div className="list-item">
            <div>
              <strong style={{ color: '#d65a2a' }}>Word-level</strong>
              <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '4px 0 0 0' }}>
                One vocabulary entry per word
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
