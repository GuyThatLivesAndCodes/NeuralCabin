import { useEffect, useRef, useState } from 'react'
import { type UnlistenFn } from '@tauri-apps/api/event'
import { inference, Network, InferenceToken } from '../api'
import type { TabProps } from '../App'

export default function InferenceTab({ network }: TabProps) {
  const [error, setError] = useState<string | null>(null)

  return (
    <div className="tab-content">
      <h2>Inference</h2>
      <p className="muted">Run a trained network on new inputs.</p>

      {error && <div className="status error">{error}</div>}

      {network && !network.trained && (
        <div className="status mt-1">
          This network hasn’t been trained yet — predictions will be random.
        </div>
      )}

      {!network ? (
        <div className="card"><p className="muted">Select a network to run inference.</p></div>
      ) : network.kind === 'feedforward' ? (
        <FeedforwardInference network={network} onError={setError} />
      ) : (
        <NextTokenInference network={network} onError={setError} />
      )}
    </div>
  )
}

// ─── Feed-forward inference ─────────────────────────────────────────────────

function FeedforwardInference({ network, onError }: {
  network: Network; onError: (e: string | null) => void
}) {
  const [inputs, setInputs] = useState<string[]>(() => Array(network.input_dim).fill('0'))
  const [output, setOutput] = useState<number[] | null>(null)
  const [busy, setBusy] = useState(false)

  // Reset when network changes
  useEffect(() => { setInputs(Array(network.input_dim).fill('0')); setOutput(null) },
    [network.id, network.input_dim])

  const setVal = (i: number, v: string) =>
    setInputs(prev => prev.map((x, idx) => idx === i ? v : x))

  const run = async () => {
    onError(null)
    const features = inputs.map(s => Number(s))
    if (features.some(n => !Number.isFinite(n))) {
      onError('All inputs must be valid numbers.')
      return
    }
    setBusy(true)
    try {
      const r = await inference.run({ network_id: network.id, features })
      setOutput(r.output ?? null)
    } catch (e) { onError(String(e)) }
    finally { setBusy(false) }
  }

  return (
    <>
      <div className="card">
        <h3>Inputs</h3>
        <div className="grid-3">
          {inputs.map((v, i) => (
            <div key={i}>
              <label>x{i}</label>
              <input value={v} onChange={e => setVal(i, e.target.value)} />
            </div>
          ))}
        </div>
        <div className="flex mt-2">
          <button onClick={run} disabled={busy}>
            {busy ? 'Running…' : 'Predict'}
          </button>
        </div>
      </div>

      {output && (
        <div className="card">
          <h3>Output</h3>
          <table>
            <thead><tr><th>Index</th><th>Value</th></tr></thead>
            <tbody>
              {output.map((v, i) => (
                <tr key={i}><td><code>y{i}</code></td><td style={{ fontVariantNumeric: 'tabular-nums' }}>{v.toFixed(6)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

// ─── Next-token inference (streaming) ───────────────────────────────────────

function NextTokenInference({ network, onError }: {
  network: Network; onError: (e: string | null) => void
}) {
  const [prompt, setPrompt] = useState<string>('')
  const [maxNewTokens, setMaxNewTokens] = useState(64)
  const [temperature, setTemperature] = useState(0.0)
  const [generated, setGenerated] = useState<string>('')
  const [tokens, setTokens] = useState<InferenceToken[]>([])
  const [busy, setBusy] = useState(false)
  const [inferenceId, setInferenceId] = useState<string | null>(null)
  const cleanupRef = useRef<UnlistenFn[]>([])

  useEffect(() => { setGenerated(''); setTokens([]); setInferenceId(null) }, [network.id])

  useEffect(() => {
    if (!inferenceId) return
    let cancelled = false
    const setup = async () => {
      const u1 = await inference.onToken(t => {
        if (cancelled || t.inference_id !== inferenceId) return
        setTokens(prev => [...prev, t])
        setGenerated(prev => prev + t.token)
      })
      const u2 = await inference.onFinished(r => {
        if (cancelled || r.inference_id !== inferenceId) return
        setBusy(false)
      })
      const u3 = await inference.onError(err => {
        if (cancelled || err.inference_id !== inferenceId) return
        onError(err.message); setBusy(false)
      })
      cleanupRef.current = [u1, u2, u3]
    }
    void setup()
    return () => { cancelled = true; cleanupRef.current.forEach(c => c()) }
  }, [inferenceId])

  const run = async () => {
    onError(null); setBusy(true); setGenerated(''); setTokens([])
    try {
      const r = await inference.run({
        network_id: network.id,
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
      })
      setInferenceId(r.inference_id ?? null)
    } catch (e) { onError(String(e)); setBusy(false) }
  }

  return (
    <>
      <div className="card">
        <h3>Prompt</h3>
        <textarea
          rows={4}
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Type a prompt to continue…"
        />
        <div className="grid-3 mt-2">
          <div>
            <label>Max new tokens</label>
            <input type="number" min={1} max={2048} value={maxNewTokens}
              onChange={e => setMaxNewTokens(Math.max(1, Math.min(2048, parseInt(e.target.value) || 1)))} />
          </div>
          <div>
            <label>Temperature</label>
            <input type="number" step="0.1" min={0} value={temperature}
              onChange={e => setTemperature(Math.max(0, parseFloat(e.target.value) || 0))} />
            <small>0 = greedy / argmax. Higher = more random.</small>
          </div>
        </div>
        <div className="flex mt-2">
          <button onClick={run} disabled={busy}>{busy ? 'Generating…' : 'Generate'}</button>
        </div>
      </div>

      {generated && (
        <div className="card">
          <h3>Output</h3>
          <div style={{
            background: 'var(--bg-input)', padding: 14, borderRadius: 'var(--radius)',
            border: '1px solid var(--border)', whiteSpace: 'pre-wrap',
            fontFamily: 'var(--font-mono)', fontSize: 13,
          }}>
            <span className="muted">{prompt}</span>
            <span style={{ color: 'var(--accent)' }}>{generated}</span>
          </div>
        </div>
      )}

      {tokens.length > 0 && (
        <div className="card">
          <h3>Per-token probabilities</h3>
          <div style={{ maxHeight: 260, overflow: 'auto' }}>
            <table>
              <thead><tr><th style={{ width: 40 }}>#</th><th>Token</th><th style={{ width: 120 }}>Probability</th></tr></thead>
              <tbody>
                {tokens.map((s, i) => (
                  <tr key={i}>
                    <td><code>{i + 1}</code></td>
                    <td><code>{visualize(s.token)}</code></td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{(s.probability * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}

function visualize(t: string): string {
  if (t === '\n') return '\\n'
  if (t === '\t') return '\\t'
  if (t === ' ')  return '·'  // visible space
  return t
}
