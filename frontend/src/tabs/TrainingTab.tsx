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
      <h2>Training</h2>

      {error && <div className="status error">{error}</div>}

      {!trainingId ? (
        <div>
          <div className="card">
            <h3>Setup Training</h3>

            <div style={{ marginBottom: '12px' }}>
              <label>Network:</label>
              {networksList.length === 0 ? (
                <p style={{ color: '#a0a0a0' }}>No networks found. Create one in the Networks tab.</p>
              ) : (
                <select
                  value={selectedNetworkId}
                  onChange={(e) => setSelectedNetworkId(e.target.value)}
                >
                  {networksList.map((net) => (
                    <option key={net.id} value={net.id}>
                      {net.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            <div style={{ marginBottom: '12px' }}>
              <label>Dataset:</label>
              {datasetsList.length === 0 ? (
                <button onClick={createDatasetXor}>Create XOR Dataset</button>
              ) : (
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                >
                  {datasetsList.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            <button onClick={startTraining} disabled={!selectedNetworkId || !selectedDatasetId}>
              Start Training
            </button>
          </div>
        </div>
      ) : (
        <div>
          <div className="status success">
            Training: Epoch {trainingStatus.epoch} / {trainingStatus.totalEpochs}
          </div>

          <div className="card">
            <h3>Training Progress</h3>
            <p>Loss: <strong>{trainingStatus.lastLoss.toFixed(6)}</strong></p>
            <p>Elapsed: {trainingStatus.elapsedSecs.toFixed(2)}s</p>
            <div
              style={{
                width: '100%',
                height: '24px',
                background: '#1a1a1a',
                borderRadius: '4px',
                overflow: 'hidden',
                marginTop: '8px',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${(trainingStatus.epoch / trainingStatus.totalEpochs) * 100}%`,
                  background: '#0066cc',
                  transition: 'width 0.2s',
                }}
              />
            </div>
          </div>

          <div className="plot">
            <h3>Loss History</h3>
            {trainingStatus.lossHistory.length > 0 ? (
              <svg
                width="100%"
                height="200"
                style={{ background: '#1a1a1a', marginTop: '8px', borderRadius: '4px' }}
              >
                <polyline
                  points={trainingStatus.lossHistory
                    .map((loss, i) => {
                      const x = (i / trainingStatus.lossHistory.length) * 100 + '%'
                      const maxLoss = Math.max(...trainingStatus.lossHistory)
                      const y = (1 - loss / maxLoss) * 180 + 10
                      return `${x} ${y}`
                    })
                    .join(' ')}
                  fill="none"
                  stroke="#0066cc"
                  strokeWidth="2"
                />
              </svg>
            ) : (
              <p style={{ color: '#a0a0a0' }}>Waiting for training data...</p>
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
            >
              Start New Training
            </button>
          )}
        </div>
      )}
    </div>
  )
}
