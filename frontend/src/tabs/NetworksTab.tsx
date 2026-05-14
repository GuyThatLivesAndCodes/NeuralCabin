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
      <h2>Networks</h2>

      {error && <div className="status error">{error}</div>}

      <button onClick={() => setShowForm(!showForm)} style={{ marginBottom: '16px' }}>
        {showForm ? 'Cancel' : 'Create Network'}
      </button>

      {showForm && (
        <div className="card">
          <h3>Create New Network</h3>
          <div style={{ display: 'grid', gap: '8px' }}>
            <div>
              <label>Name:</label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>
            <div>
              <label>Kind:</label>
              <select
                value={formData.kind}
                onChange={(e) => setFormData({ ...formData, kind: e.target.value })}
              >
                <option>simplex</option>
                <option>gpt</option>
                <option>next_token_gen</option>
              </select>
            </div>
            <div>
              <label>Seed:</label>
              <input
                type="number"
                value={formData.seed}
                onChange={(e) => setFormData({ ...formData, seed: parseInt(e.target.value) })}
              />
            </div>
            <button onClick={createNetwork} disabled={loading}>
              {loading ? 'Creating...' : 'Create'}
            </button>
          </div>
        </div>
      )}

      <h3>Existing Networks ({networksList.length})</h3>
      {networksList.length === 0 ? (
        <p style={{ color: '#a0a0a0' }}>No networks yet. Create one to get started!</p>
      ) : (
        networksList.map((network) => (
          <div key={network.id} className="list-item">
            <div>
              <strong>{network.name}</strong>
              <p style={{ fontSize: '12px', color: '#808080', margin: '4px 0 0 0' }}>
                {network.kind} • {network.layers.length} layers • Seed: {network.seed}
              </p>
            </div>
            <button onClick={() => deleteNetwork(network.id)}>Delete</button>
          </div>
        ))
      )}
    </div>
  )
}
