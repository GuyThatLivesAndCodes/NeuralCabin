import { useEffect, useRef, useState } from 'react'
import { type UnlistenFn } from '@tauri-apps/api/event'
import { inference, vocabulary, Network, InferenceToken } from '../api'
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

// UNK_ID — matches `tokenizer::UNK_ID` in the engine. The backend silently
// emits this for any character/word that isn't in the trained vocabulary, so
// we have to detect it explicitly in the frontend to fail fast.
const UNK_ID = 1

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
  // Refs let the event listeners read the latest values without forcing a
  // re-subscribe on every state change. Critical for the race fix: listeners
  // are attached ONCE on mount, before the user can click Generate.
  const activeIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)

  useEffect(() => {
    setGenerated(''); setTokens([]); setInferenceId(null)
    activeIdRef.current = null
    busyRef.current = false
    setBusy(false)
  }, [network.id])

  // Attach listeners ONCE per network. Previously these were keyed on
  // `inferenceId`, so they only got registered AFTER `inference.run()`
  // returned — by which point the backend had already emitted events
  // (sometimes including `inference_finished`), and we missed them. Missing
  // `inference_finished` is exactly what made the Generate button stay
  // grayed out forever.
  useEffect(() => {
    let cancelled = false
    const cleanups: UnlistenFn[] = []
    const setup = async () => {
      const u1 = await inference.onToken(t => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && t.inference_id !== activeIdRef.current) return
        setTokens(prev => [...prev, t])
        setGenerated(prev => prev + t.token)
      })
      const u2 = await inference.onFinished(r => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && r.inference_id !== activeIdRef.current) return
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      const u3 = await inference.onError(err => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && err.inference_id !== activeIdRef.current) return
        onError(err.message)
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      if (cancelled) { u1(); u2(); u3(); return }
      cleanups.push(u1, u2, u3)
    }
    void setup()
    return () => { cancelled = true; cleanups.forEach(c => c()) }
  }, [network.id])

  const run = async () => {
    onError(null)
    setGenerated(''); setTokens([])
    // Mark busy BEFORE awaiting anything so the always-on listeners will
    // accept the events that follow.
    busyRef.current = true
    activeIdRef.current = null
    setBusy(true)
    try {
      if (prompt) {
        const toks = await vocabulary.tokenize(network.id, prompt)
        const unknown = toks.filter(([id]) => id === UNK_ID)
        if (unknown.length > 0) {
          const sample = unknown.slice(0, 5).map(([, s]) => JSON.stringify(s)).join(', ')
          throw new Error(
            `Prompt contains ${unknown.length} token(s) not in the trained vocabulary` +
            ` (e.g. ${sample}). Train on text that includes these characters first,` +
            ` or remove them from the prompt.`
          )
        }
      }
      const r = await inference.run({
        network_id: network.id,
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
      })
      const id = r.inference_id ?? null
      activeIdRef.current = id
      setInferenceId(id)
      if (!id) {
        // Synchronous result (shouldn't happen for next-token, but bail safely
        // rather than leaving the button stuck on "Generating…").
        busyRef.current = false
        setBusy(false)
      }
    } catch (e) {
      busyRef.current = false
      activeIdRef.current = null
      setBusy(false)
      onError(String(e))
    }
  }

  const stop = async () => {
    const id = activeIdRef.current ?? inferenceId
    if (!id) {
      // No active inference but the UI is stuck — unstick it locally.
      busyRef.current = false
      setBusy(false)
      setInferenceId(null)
      return
    }
    try {
      await inference.stop(id)
    } catch (e) {
      onError(String(e))
    }
    // Belt-and-braces: if the backend never emits `inference_finished` within
    // 2 seconds of a stop, force the UI back to an idle state so the Generate
    // button is usable again.
    setTimeout(() => {
      if (busyRef.current && activeIdRef.current === id) {
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      }
    }, 2000)
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
          {busy && (
            <button className="secondary" onClick={stop}>
              Stop
            </button>
          )}
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
