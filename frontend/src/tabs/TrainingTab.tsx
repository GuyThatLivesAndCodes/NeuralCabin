import { useEffect, useRef, useState } from 'react'
import { type UnlistenFn } from '@tauri-apps/api/event'
import {
  training, corpus, OptimizerConfig, CorpusStats, TrainingRun,
} from '../api'
import type { TabProps } from '../App'

interface RunState {
  running: boolean
  epoch: number
  totalEpochs: number
  lastLoss: number
  lossHistory: number[]
  elapsedSecs: number
  finalStatus?: 'completed' | 'cancelled' | 'aborted'
}

const EMPTY_RUN: RunState = { running: false, epoch: 0, totalEpochs: 0, lastLoss: 0, lossHistory: [], elapsedSecs: 0 }

export default function TrainingTab({ network, refreshNetworks }: TabProps) {
  const [stats, setStats] = useState<CorpusStats | null>(null)

  const [epochs, setEpochs] = useState(500)
  const [batchSize, setBatchSize] = useState(32)
  const [seed, setSeed] = useState(42)
  const [optKind, setOptKind] = useState<'adam' | 'adamw' | 'lamb' | 'sgd'>('adam')
  const [lr, setLr] = useState(0.01)
  const [momentum, setMomentum] = useState(0.9)
  const [maskUserTokens, setMaskUserTokens] = useState(true)

  const [trainingId, setTrainingId] = useState<string | null>(null)
  const [run, setRun] = useState<RunState>(EMPTY_RUN)
  const [error, setError] = useState<string | null>(null)
  const [historyVersion, setHistoryVersion] = useState(0)
  const finishedListenersRef = useRef<UnlistenFn[]>([])

  useEffect(() => {
    if (network) void loadStats(network.id)
    else setStats(null)
  }, [network])

  const isNextToken = network?.kind === 'next_token'
  const lossLabel = isNextToken ? 'crossentropy (forced for next-token)' : 'mse'

  // Subscribe to training events whenever a job is active
  useEffect(() => {
    if (!trainingId) return
    let cancelled = false
    const cleanups: UnlistenFn[] = []

    const setup = async () => {
      const u = await training.onUpdate(u => {
        if (cancelled || u.training_id !== trainingId) return
        setRun({
          running: true, epoch: u.epoch, totalEpochs: u.total_epochs,
          lastLoss: u.loss, lossHistory: u.loss_history, elapsedSecs: u.elapsed_secs,
        })
      })
      const f = await training.onFinished(r => {
        if (cancelled || r.training_id !== trainingId) return
        setRun(prev => ({
          ...prev,
          running: false,
          lastLoss: r.final_loss,
          elapsedSecs: r.elapsed_secs,
          finalStatus: r.status,
        }))
        // Reload network list so "trained" badge updates, and bump history.
        void refreshNetworks()
        setHistoryVersion(v => v + 1)
      })
      const e = await training.onError(err => {
        if (cancelled || err.training_id !== trainingId) return
        setError(err.message)
        setRun(prev => ({ ...prev, running: false }))
        setHistoryVersion(v => v + 1)
      })
      cleanups.push(u, f, e)
    }
    void setup()
    finishedListenersRef.current = cleanups
    return () => { cancelled = true; cleanups.forEach(c => c()) }
  }, [trainingId])

  const loadStats = async (id: string) => {
    try { setStats(await corpus.stats(id)) }
    catch (e) { setStats(null) /* not having a corpus is OK */ }
  }

  const start = async () => {
    if (!network) return
    setError(null); setRun({ ...EMPTY_RUN, running: true, totalEpochs: epochs })
    const optimizer: OptimizerConfig = { kind: optKind, lr }
    if (optKind === 'sgd') optimizer.momentum = momentum
    try {
      const r = await training.start({
        network_id: network.id,
        config: {
          epochs, batch_size: batchSize, optimizer,
          loss: isNextToken ? 'crossentropy' : 'mse',
          seed,
          mask_user_tokens: isNextToken ? maskUserTokens : undefined,
        },
      })
      setTrainingId(r.training_id)
    } catch (e) {
      setError(String(e))
      setRun(EMPTY_RUN)
    }
  }

  const reset = () => { setTrainingId(null); setRun(EMPTY_RUN); setError(null) }

  const examples = stats?.training_examples ?? 0
  const corpusReady = examples > 0

  return (
    <div className="tab-content">
      <h2>Training</h2>
      <p className="muted">Train the selected network on its attached corpus.</p>

      {error && <div className="status error">{error}</div>}

      {network && (
        <div className="card">
          <p className="muted">
            <span className="chip">{network.name}</span>{' '}
            <span className="chip">{network.input_dim} → {network.output_dim}</span>{' '}
            <span className="chip">{network.parameter_count.toLocaleString()} params</span>{' '}
            <span className="chip">{examples.toLocaleString()} training examples</span>
            {!corpusReady && <span className="status error" style={{ display: 'inline-block', marginLeft: 8 }}>
              No corpus attached. Add data on the Corpus tab.
            </span>}
          </p>
        </div>
      )}

      {network && !trainingId && (
        <HistoryPanel networkId={network.id} refreshKey={historyVersion} />
      )}

      {!trainingId ? (
        <div className="card">
          <h3>Configuration</h3>
          <div className="grid-3">
            <div>
              <label>Epochs</label>
              <input type="number" min={1} value={epochs}
                onChange={e => setEpochs(Math.max(1, parseInt(e.target.value) || 1))} />
            </div>
            <div>
              <label>Batch size</label>
              <input type="number" min={1} value={batchSize}
                onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))} />
            </div>
            <div>
              <label>Seed</label>
              <input type="number" value={seed}
                onChange={e => setSeed(parseInt(e.target.value) || 0)} />
            </div>
            <div>
              <label>Optimizer</label>
              <select value={optKind} onChange={e => setOptKind(e.target.value as 'adam' | 'adamw' | 'lamb' | 'sgd')}>
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
                <option value="lamb">LAMB</option>
                <option value="sgd">SGD</option>
              </select>
            </div>
            <div>
              <label>Learning rate</label>
              <input type="number" step="0.001" min={0} value={lr}
                onChange={e => setLr(parseFloat(e.target.value) || 0)} />
            </div>
            {optKind === 'sgd' && (
              <div>
                <label>Momentum</label>
                <input type="number" step="0.01" min={0} max={1} value={momentum}
                  onChange={e => setMomentum(parseFloat(e.target.value) || 0)} />
              </div>
            )}
            <div>
              <label>Loss</label>
              <input value={lossLabel} disabled />
            </div>
            {isNextToken && stats?.stage === 'finetune' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <label>Mask user tokens</label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, textTransform: 'none', letterSpacing: 0, color: 'var(--text)' }}>
                  <input type="checkbox" style={{ width: 'auto' }} checked={maskUserTokens}
                    onChange={e => setMaskUserTokens(e.target.checked)} />
                  Score loss only on assistant output
                </label>
              </div>
            )}
          </div>
          <div className="flex mt-2">
            <button onClick={start} disabled={!network || !corpusReady}>
              Start training
            </button>
          </div>
        </div>
      ) : (
        <RunView run={run} trainingId={trainingId} onReset={reset} onError={setError} />
      )}
    </div>
  )
}

function RunView({ run, trainingId, onReset, onError }: {
  run: RunState
  trainingId: string | null
  onReset: () => void
  onError: (e: string | null) => void
}) {
  const [stopping, setStopping] = useState<'stop' | 'abort' | null>(null)

  const onStopHere = async () => {
    if (!trainingId) return
    setStopping('stop'); onError(null)
    try { await training.stop(trainingId) }
    catch (e) { onError(String(e)) }
  }
  const onAbort = async () => {
    if (!trainingId) return
    setStopping('abort'); onError(null)
    try { await training.abort(trainingId) }
    catch (e) { onError(String(e)) }
  }

  return (
    <>
      <div className={`status ${run.running ? '' : (run.finalStatus === 'aborted' ? 'error' : 'success')}`}>
        {run.running
          ? `Training… epoch ${run.epoch} / ${run.totalEpochs}`
          : run.finalStatus === 'aborted'
            ? `Aborted — model rolled back to pre-training weights (${run.elapsedSecs.toFixed(1)}s)`
            : run.finalStatus === 'cancelled'
              ? `Stopped at epoch ${run.epoch} — kept current weights (${run.elapsedSecs.toFixed(1)}s)`
              : `Done — ${run.totalEpochs} epochs in ${run.elapsedSecs.toFixed(1)}s`}
      </div>

      <div className="card">
        <div className="grid-3">
          <Metric label="Epoch" value={`${run.epoch} / ${run.totalEpochs}`} />
          <Metric label="Loss"  value={run.lastLoss.toFixed(6)} />
          <Metric label="Time"  value={`${run.elapsedSecs.toFixed(1)}s`} />
        </div>
        <ProgressBar epoch={run.epoch} total={run.totalEpochs} />

        {run.running && (
          <div className="flex mt-2">
            <button
              className="secondary"
              onClick={onStopHere}
              disabled={stopping !== null}
              title="Halt training and keep whatever the model has learned so far."
            >
              {stopping === 'stop' ? 'Stopping…' : 'Stop here'}
            </button>
            <button
              className="danger"
              onClick={onAbort}
              disabled={stopping !== null}
              title="Halt training AND revert the model to its pre-training weights."
            >
              {stopping === 'abort' ? 'Aborting…' : 'Abort'}
            </button>
          </div>
        )}
      </div>

      <div className="plot">
        <h3>Loss over epochs</h3>
        <LossPlot history={run.lossHistory} />
      </div>

      {!run.running && (
        <button onClick={onReset}>Start a new run</button>
      )}
    </>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="muted small">{label}</p>
      <p style={{ fontSize: 22, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{value}</p>
    </div>
  )
}

function ProgressBar({ epoch, total }: { epoch: number; total: number }) {
  const pct = total > 0 ? (epoch / total) * 100 : 0
  return (
    <div style={{ marginTop: 14 }}>
      <div style={{ height: 6, background: 'var(--bg-input)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'var(--accent)', transition: 'width 0.2s ease' }} />
      </div>
      <p className="muted small mt-1" style={{ textAlign: 'right' }}>{pct.toFixed(1)}%</p>
    </div>
  )
}

function LossPlot({ history, color = 'var(--accent)' }: { history: number[]; color?: string }) {
  if (history.length === 0) {
    return <p className="muted">Waiting for first epoch…</p>
  }
  const max = Math.max(...history)
  const min = Math.min(...history)
  const range = max - min || 1

  // SVG viewBox in 100x100, paint grid + line
  const points = history.map((v, i) => {
    const x = (i / Math.max(history.length - 1, 1)) * 100
    const y = 100 - ((v - min) / range) * 90 - 5
    return `${x},${y}`
  }).join(' ')

  return (
    <div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none"
           style={{ width: '100%', height: 220, background: 'var(--bg-input)', borderRadius: 4 }}>
        {[25, 50, 75].map(y => (
          <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="var(--border-soft)" strokeWidth="0.2" />
        ))}
        <polyline points={points} fill="none" stroke={color} strokeWidth="1" vectorEffect="non-scaling-stroke" />
      </svg>
      <p className="muted small mt-1">
        min {min.toFixed(6)} · max {max.toFixed(6)} · last {history[history.length - 1].toFixed(6)}
      </p>
    </div>
  )
}

// ─── Training history panel ──────────────────────────────────────────────────

function HistoryPanel({ networkId, refreshKey }: { networkId: string; refreshKey: number }) {
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)

  useEffect(() => {
    training.history(networkId).then(setRuns).catch(() => setRuns([]))
  }, [networkId, refreshKey])

  if (runs.length === 0) return null

  const clearAll = async () => {
    setClearing(true)
    try {
      await training.clearHistory(networkId)
      setRuns([])
      setExpanded(null)
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="card">
      <div className="card-row" style={{ marginBottom: 12 }}>
        <h3 style={{ marginBottom: 0 }}>Training History</h3>
        <button className="ghost" style={{ fontSize: 12, padding: '4px 10px' }}
          onClick={clearAll} disabled={clearing}>
          {clearing ? 'Clearing…' : 'Clear all'}
        </button>
      </div>
      {runs.map(run => (
        <HistoryRunItem
          key={run.id}
          run={run}
          expanded={expanded === run.id}
          onToggle={() => setExpanded(prev => prev === run.id ? null : run.id)}
        />
      ))}
    </div>
  )
}

const STATUS_COLORS: Record<string, string> = {
  completed: 'var(--success)',
  cancelled:  'var(--text-muted)',
  aborted:    'var(--error)',
  error:      'var(--error)',
}

function HistoryRunItem({ run, expanded, onToggle }: {
  run: TrainingRun
  expanded: boolean
  onToggle: () => void
}) {
  const date = new Date(run.started_at)
  const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  const statusColor = STATUS_COLORS[run.status] ?? 'var(--text-muted)'
  const plotColor = run.status === 'completed' ? 'var(--accent)' : 'var(--text-faint)'

  return (
    <div style={{ marginBottom: 6 }}>
      <button
        onClick={onToggle}
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          width: '100%', textAlign: 'left',
          background: expanded ? 'var(--bg-elev-2)' : 'transparent',
          border: '1px solid var(--border-soft)',
          borderRadius: 'var(--radius)',
          padding: '8px 12px',
          cursor: 'pointer',
          color: 'var(--text)',
          fontFamily: 'var(--font-sans)',
          fontSize: 13,
          transition: 'background 0.12s ease',
        }}
      >
        <span style={{ color: statusColor, fontSize: 11, fontWeight: 600,
                        textTransform: 'uppercase', letterSpacing: '0.06em', minWidth: 64 }}>
          {run.status}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 12, minWidth: 90 }}>
          {dateStr} {timeStr}
        </span>
        <span className="chip" style={{ fontSize: 11 }}>
          {run.config_summary.optimizer} lr={run.config_summary.lr}
        </span>
        <span className="chip" style={{ fontSize: 11 }}>
          {run.epochs_run}/{run.total_epochs} epochs
        </span>
        <span className="chip" style={{ fontSize: 11 }}>
          loss {run.final_loss.toFixed(6)}
        </span>
        <span style={{ color: 'var(--text-faint)', fontSize: 12, marginLeft: 'auto' }}>
          {run.elapsed_secs.toFixed(1)}s
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          {expanded ? '▲' : '▼'}
        </span>
      </button>
      {expanded && run.loss_history.length > 0 && (
        <div style={{ padding: '12px 12px 4px', background: 'var(--bg-elev-2)',
                      border: '1px solid var(--border-soft)', borderTop: 'none',
                      borderRadius: '0 0 var(--radius) var(--radius)' }}>
          <LossPlot history={run.loss_history} color={plotColor} />
        </div>
      )}
    </div>
  )
}
