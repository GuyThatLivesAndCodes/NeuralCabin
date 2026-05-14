import { useState, useEffect } from 'react'
import { networks, Network, Layer } from '../api'

export default function NetworksTab() {
  const [networksList, setNetworksList] = useState<Network[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({
    name: 'xor-mlp',
    kind: 'simplex',
    seed: 42,
  })

  useEffect(() => {
    loadNetworks()
  }, [])

  const loadNetworks = async () => {
    setLoading(true)
    try {
      const response = await networks.list()
      setNetworksList(response.data.networks)
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const createNetwork = async () => {
    setLoading(true)
    try {
      const layers: Layer[] = [
        { type: 'linear', in_dim: 2, out_dim: 8 },
        { type: 'activation', activation: 'tanh' },
        { type: 'linear', in_dim: 8, out_dim: 1 },
        { type: 'activation', activation: 'sigmoid' },
      ]

      await networks.create({
        name: formData.name,
        kind: formData.kind,
        seed: formData.seed,
        layers,
      })

      await loadNetworks()
      setShowForm(false)
      setFormData({ name: 'xor-mlp', kind: 'simplex', seed: 42 })
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const deleteNetwork = async (id: string) => {
    try {
      await networks.delete(id)
      await loadNetworks()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="tab-content">
      <h2>🧠 Neural Networks</h2>
      <p style={{ color: '#7d6b5f', marginBottom: '20px' }}>
        Create and manage neural network architectures. Each network can be trained with different datasets.
      </p>

      {error && <div className="status error">{error}</div>}

      <button
        onClick={() => setShowForm(!showForm)}
        style={{ marginBottom: '20px', width: '100%', padding: '12px', fontSize: '16px' }}
      >
        {showForm ? '✕ Cancel' : '+ Create New Network'}
      </button>

      {showForm && (
        <div className="card">
          <h3>Create New Network</h3>
          <p style={{ color: '#7d6b5f', marginBottom: '16px' }}>
            Configure the architecture of your neural network.
          </p>
          <div style={{ display: 'grid', gap: '16px' }}>
            <div>
              <label>Network Name:</label>
              <input
                type="text"
                placeholder="e.g., my-xor-network"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>
            <div>
              <label>Network Type:</label>
              <select
                value={formData.kind}
                onChange={(e) => setFormData({ ...formData, kind: e.target.value })}
              >
                <option value="simplex">Simplex (Feed-forward)</option>
                <option value="gpt">GPT-style</option>
                <option value="next_token_gen">Next Token Generator</option>
              </select>
            </div>
            <div>
              <label>Random Seed:</label>
              <input
                type="number"
                value={formData.seed}
                onChange={(e) => setFormData({ ...formData, seed: parseInt(e.target.value) })}
              />
              <p style={{ fontSize: '12px', color: '#9b8a7f', marginTop: '4px' }}>
                For reproducibility of weight initialization
              </p>
            </div>
            <button onClick={createNetwork} disabled={loading} style={{ width: '100%', padding: '12px' }}>
              {loading ? '⏳ Creating...' : '✓ Create Network'}
            </button>
          </div>
        </div>
      )}

      <h3>
        {networksList.length === 0 ? 'No Networks Yet' : `Your Networks (${networksList.length})`}
      </h3>
      {networksList.length === 0 ? (
        <div className="card">
          <p style={{ color: '#9b8a7f', textAlign: 'center', padding: '20px 0' }}>
            No networks created yet. Click the button above to create your first network!
          </p>
        </div>
      ) : (
        <div>
          {networksList.map((network) => (
            <div key={network.id} className="list-item">
              <div style={{ flex: 1 }}>
                <strong>🧠 {network.name}</strong>
                <p style={{ fontSize: '13px', color: '#9b8a7f', margin: '6px 0 0 0' }}>
                  <span style={{ color: '#d65a2a', fontWeight: '600' }}>Type:</span> {network.kind} •
                  <span style={{ color: '#d65a2a', fontWeight: '600' }}> Layers:</span> {network.layers.length} •
                  <span style={{ color: '#d65a2a', fontWeight: '600' }}> Seed:</span> {network.seed}
                </p>
              </div>
              <button onClick={() => deleteNetwork(network.id)} style={{ fontSize: '13px' }}>
                🗑 Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
