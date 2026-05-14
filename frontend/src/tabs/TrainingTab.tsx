import { useState, useEffect, useRef } from 'react'
import { networks, datasets, training, Network, Dataset, WsMessage } from '../api'

export default function TrainingTab() {
  const [networksList, setNetworksList] = useState<Network[]>([])
  const [datasetsList, setDatasetsList] = useState<Dataset[]>([])
  const [selectedNetworkId, setSelectedNetworkId] = useState('')
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [trainingId, setTrainingId] = useState<string | null>(null)
  const [trainingStatus, setTrainingStatus] = useState({
    running: false,
    epoch: 0,
    totalEpochs: 0,
    lastLoss: 0,
    lossHistory: [] as number[],
    elapsedSecs: 0,
  })
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    loadNetworks()
    loadDatasets()
  }, [])

  useEffect(() => {
    if (!trainingId) return

    const ws = training.connect(trainingId)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)

        if (msg.type === 'epoch_update') {
          setTrainingStatus({
            running: true,
            epoch: msg.epoch || 0,
            totalEpochs: msg.total_epochs || 0,
            lastLoss: msg.last_loss || 0,
            lossHistory: msg.loss_history || [],
            elapsedSecs: msg.elapsed_secs || 0,
          })
        } else if (msg.type === 'training_finished') {
          setTrainingStatus((prev) => ({
            ...prev,
            running: false,
          }))
        } else if (msg.type === 'error') {
          setError(msg.message || 'Unknown error')
          setTrainingStatus((prev) => ({
            ...prev,
            running: false,
          }))
        }
      } catch (e) {
        console.error('Failed to parse WS message:', e)
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection error')
    }

    return () => {
      ws.close()
    }
  }, [trainingId])

  const loadNetworks = async () => {
    try {
      const response = await networks.list()
      setNetworksList(response.data.networks)
      if (response.data.networks.length > 0) {
        setSelectedNetworkId(response.data.networks[0].id)
      }
    } catch (e) {
      setError(String(e))
    }
  }

  const loadDatasets = async () => {
    try {
      const response = await datasets.list()
      setDatasetsList(response.data.datasets)
      if (response.data.datasets.length > 0) {
        setSelectedDatasetId(response.data.datasets[0].id)
      }
    } catch (e) {
      setError(String(e))
    }
  }

  const createDatasetXor = async () => {
    try {
      await datasets.create({ name: 'xor', kind: 'xor', seed: 42 })
      await loadDatasets()
    } catch (e) {
      setError(String(e))
    }
  }

  const startTraining = async () => {
    if (!selectedNetworkId || !selectedDatasetId) {
      setError('Please select both network and dataset')
      return
    }

    try {
      const response = await training.start({
        network_id: selectedNetworkId,
        dataset_id: selectedDatasetId,
        config: {
          epochs: 2000,
          batch_size: 4,
          optimizer: {
            kind: 'adam',
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
          },
          loss: 'mse',
          validation_frac: 0.2,
          seed: 42,
        },
      })

      setTrainingId(response.data.training_id)
      setTrainingStatus({
        running: true,
        epoch: 0,
        totalEpochs: 2000,
        lastLoss: 0,
        lossHistory: [],
        elapsedSecs: 0,
      })
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="tab-content">
      <h2>⚡ Training</h2>

      {error && <div className="status error">{error}</div>}

      {!trainingId ? (
        <div>
          <div className="card">
            <h3>Configure Training Session</h3>
            <p style={{ color: '#7d6b5f', marginBottom: '16px' }}>
              Select a network and dataset, then start training to monitor real-time loss.
            </p>

            <div style={{ marginBottom: '16px' }}>
              <label>Select Network:</label>
              {networksList.length === 0 ? (
                <p style={{ color: '#9b8a7f', fontStyle: 'italic' }}>No networks found. Create one in the Networks tab.</p>
              ) : (
                <select
                  value={selectedNetworkId}
                  onChange={(e) => setSelectedNetworkId(e.target.value)}
                >
                  {networksList.map((net) => (
                    <option key={net.id} value={net.id}>
                      {net.name} ({net.kind})
                    </option>
                  ))}
                </select>
              )}
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label>Select Dataset:</label>
              {datasetsList.length === 0 ? (
                <div>
                  <p style={{ color: '#9b8a7f', fontStyle: 'italic', marginBottom: '8px' }}>No datasets available. Create one:</p>
                  <button onClick={createDatasetXor} style={{ width: '100%' }}>+ Create XOR Dataset</button>
                </div>
              ) : (
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                >
                  {datasetsList.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} ({ds.samples} samples)
                    </option>
                  ))}
                </select>
              )}
            </div>

            <button
              onClick={startTraining}
              disabled={!selectedNetworkId || !selectedDatasetId}
              style={{ width: '100%', padding: '12px', fontSize: '16px' }}
            >
              🚀 Start Training
            </button>
          </div>

          <div className="card">
            <h3>Training Configuration</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <p style={{ color: '#d65a2a', fontWeight: '600', marginBottom: '4px' }}>Epochs</p>
                <p style={{ fontSize: '18px', fontWeight: 'bold' }}>2000</p>
              </div>
              <div>
                <p style={{ color: '#d65a2a', fontWeight: '600', marginBottom: '4px' }}>Batch Size</p>
                <p style={{ fontSize: '18px', fontWeight: 'bold' }}>4</p>
              </div>
              <div>
                <p style={{ color: '#d65a2a', fontWeight: '600', marginBottom: '4px' }}>Optimizer</p>
                <p style={{ fontSize: '14px' }}>Adam (lr=0.05)</p>
              </div>
              <div>
                <p style={{ color: '#d65a2a', fontWeight: '600', marginBottom: '4px' }}>Loss Function</p>
                <p style={{ fontSize: '14px' }}>Mean Squared Error</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <div className="status success">
            ⚡ Training in Progress: Epoch {trainingStatus.epoch} / {trainingStatus.totalEpochs}
          </div>

          <div className="card">
            <h3>Training Metrics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
              <div>
                <p style={{ color: '#9b8a7f', marginBottom: '4px' }}>Current Loss</p>
                <p style={{ fontSize: '28px', color: '#d65a2a', fontWeight: 'bold' }}>
                  {trainingStatus.lastLoss.toFixed(6)}
                </p>
              </div>
              <div>
                <p style={{ color: '#9b8a7f', marginBottom: '4px' }}>Elapsed Time</p>
                <p style={{ fontSize: '28px', color: '#d65a2a', fontWeight: 'bold' }}>
                  {trainingStatus.elapsedSecs.toFixed(1)}s
                </p>
              </div>
            </div>

            <p style={{ color: '#9b8a7f', marginBottom: '8px', fontSize: '14px' }}>Progress</p>
            <div
              style={{
                width: '100%',
                height: '32px',
                background: '#f0e6dc',
                borderRadius: '8px',
                overflow: 'hidden',
                border: '2px solid #e0b8a0',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${(trainingStatus.epoch / trainingStatus.totalEpochs) * 100}%`,
                  background: 'linear-gradient(90deg, #d65a2a 0%, #e07640 100%)',
                  transition: 'width 0.3s ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '12px',
                  fontWeight: 'bold',
                }}
              >
                {((trainingStatus.epoch / trainingStatus.totalEpochs) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="plot">
            <h3>Loss Over Time</h3>
            {trainingStatus.lossHistory.length > 0 ? (
              <svg
                width="100%"
                height="250"
                style={{ background: '#f9f5f0', marginTop: '12px', borderRadius: '6px', border: '1px solid #e0b8a0' }}
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
              >
                {/* Grid lines */}
                <line x1="0" y1="50" x2="100" y2="50" stroke="#e0b8a0" strokeWidth="0.2" />
                <line x1="0" y1="25" x2="100" y2="25" stroke="#e0b8a0" strokeWidth="0.2" />
                <line x1="0" y1="75" x2="100" y2="75" stroke="#e0b8a0" strokeWidth="0.2" />

                {/* Loss curve */}
                <polyline
                  points={trainingStatus.lossHistory
                    .map((loss, i) => {
                      const x = (i / (trainingStatus.lossHistory.length - 1 || 1)) * 100
                      const maxLoss = Math.max(...trainingStatus.lossHistory)
                      const y = (1 - loss / (maxLoss || 1)) * 90 + 5
                      return `${x} ${y}`
                    })
                    .join(' ')}
                  fill="none"
                  stroke="#d65a2a"
                  strokeWidth="0.8"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            ) : (
              <p style={{ color: '#9b8a7f', fontStyle: 'italic', marginTop: '20px', textAlign: 'center' }}>
                Waiting for training data...
              </p>
            )}
          </div>

          {!trainingStatus.running && (
            <button
              onClick={() => {
                setTrainingId(null)
                setTrainingStatus({
                  running: false,
                  epoch: 0,
                  totalEpochs: 0,
                  lastLoss: 0,
                  lossHistory: [],
                  elapsedSecs: 0,
                })
              }}
              style={{ width: '100%', padding: '12px', fontSize: '16px', marginTop: '12px' }}
            >
              🚀 Start New Training
            </button>
          )}
        </div>
      )}
    </div>
  )
}
