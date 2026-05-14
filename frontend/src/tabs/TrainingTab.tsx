import { useState, useEffect } from 'react'
import { type UnlistenFn } from '@tauri-apps/api/event'
import { networks, datasets, training, Network, Dataset } from '../api'

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

  useEffect(() => {
    loadNetworks()
    loadDatasets()
  }, [])

  useEffect(() => {
    if (!trainingId) return

    let unlistenUpdate: UnlistenFn | null = null
    let unlistenFinished: UnlistenFn | null = null
    let unlistenError: UnlistenFn | null = null

    const setup = async () => {
      unlistenUpdate = await training.onUpdate((update) => {
        if (update.training_id !== trainingId) return
        setTrainingStatus({
          running: true,
          epoch: update.epoch,
          totalEpochs: update.total_epochs,
          lastLoss: update.loss,
          lossHistory: update.loss_history,
          elapsedSecs: update.elapsed_secs,
        })
      })

      unlistenFinished = await training.onFinished((result) => {
        if (result.training_id !== trainingId) return
        setTrainingStatus((prev) => ({
          ...prev,
          running: false,
          lastLoss: result.final_loss,
        }))
      })

      unlistenError = await training.onError((err) => {
        if (err.training_id !== trainingId) return
        setError(err.message)
        setTrainingStatus((prev) => ({ ...prev, running: false }))
      })
    }

    setup()

    return () => {
      unlistenUpdate?.()
      unlistenFinished?.()
      unlistenError?.()
    }
  }, [trainingId])

  const loadNetworks = async () => {
    try {
      const list = await networks.list()
      setNetworksList(list)
      if (list.length > 0) setSelectedNetworkId(list[0].id)
    } catch (e) {
      setError(String(e))
    }
  }

  const loadDatasets = async () => {
    try {
      const list = await datasets.list()
      setDatasetsList(list)
      if (list.length > 0) setSelectedDatasetId(list[0].id)
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
          optimizer: { kind: 'adam', lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
          loss: 'mse',
          validation_frac: 0.2,
          seed: 42,
        },
      })

      setTrainingId(response.training_id)
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
                <p style={{ color: '#9b8a7f', fontStyle: 'italic' }}>
                  No networks found. Create one in the Networks tab.
                </p>
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
                  <p style={{ color: '#9b8a7f', fontStyle: 'italic', marginBottom: '8px' }}>
                    No datasets available. Create one:
                  </p>
                  <button onClick={createDatasetXor} style={{ width: '100%' }}>
                    + Create XOR Dataset
                  </button>
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
              {[
                ['Epochs', '2000'],
                ['Batch Size', '4'],
                ['Optimizer', 'Adam (lr=0.05)'],
                ['Loss Function', 'Mean Squared Error'],
              ].map(([label, value]) => (
                <div key={label}>
                  <p style={{ color: '#d65a2a', fontWeight: '600', marginBottom: '4px' }}>{label}</p>
                  <p style={{ fontSize: label === 'Epochs' || label === 'Batch Size' ? '18px' : '14px', fontWeight: 'bold' }}>
                    {value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div>
          <div className="status success">
            ⚡ Training{trainingStatus.running ? ' in Progress' : ' Complete'}: Epoch{' '}
            {trainingStatus.epoch} / {trainingStatus.totalEpochs}
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
                  width: `${(trainingStatus.epoch / (trainingStatus.totalEpochs || 1)) * 100}%`,
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
                {((trainingStatus.epoch / (trainingStatus.totalEpochs || 1)) * 100).toFixed(1)}%
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
                <line x1="0" y1="50" x2="100" y2="50" stroke="#e0b8a0" strokeWidth="0.2" />
                <line x1="0" y1="25" x2="100" y2="25" stroke="#e0b8a0" strokeWidth="0.2" />
                <line x1="0" y1="75" x2="100" y2="75" stroke="#e0b8a0" strokeWidth="0.2" />
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
                setTrainingStatus({ running: false, epoch: 0, totalEpochs: 0, lastLoss: 0, lossHistory: [], elapsedSecs: 0 })
                setError(null)
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
