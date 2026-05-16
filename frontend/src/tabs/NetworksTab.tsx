import { useEffect, useMemo, useState } from 'react'
import { networks, exporter, Activation, ExportFormat, Layer, Network, NetworkKind } from '../api'
import type { TabProps } from '../App'

const ACTIVATIONS: Activation[] = ['identity', 'relu', 'sigmoid', 'tanh', 'softmax']

interface FormState {
  name: string
  kind: NetworkKind
  seed: number
  // Feedforward fields
  inputDim: number
  hiddenSpec: string  // e.g. "8,relu,4,relu" — comma list of (number | activation)
  outputDim: number
  outputActivation: Activation
  // Next-token fields
  vocabBudget: number  // Approx vocab size you expect — used to size the input layer
  contextSize: number
  embedHidden: string  // hidden layer sizes after the one-hot input, e.g. "64,relu,32,relu"
}

function defaultForm(kind: NetworkKind): FormState {
  if (kind === 'feedforward') {
    return {
      name: 'xor-mlp', kind, seed: 42,
      inputDim: 2, hiddenSpec: '8,tanh', outputDim: 1, outputActivation: 'sigmoid',
      vocabBudget: 64, contextSize: 16, embedHidden: '64,relu',
    }
  }
  return {
    name: 'tiny-lm', kind, seed: 42,
    inputDim: 2, hiddenSpec: '8,tanh', outputDim: 1, outputActivation: 'sigmoid',
    vocabBudget: 64, contextSize: 16, embedHidden: '64,relu',
  }
}

/**
 * Parse a hidden-layer spec like "8,relu,4,relu" into a sequence of
 * (linearOutDim | activation) tokens. Returns null + error on parse failure.
 */
function parseSpec(spec: string): { tokens: Array<number | Activation>; error?: string } {
  const parts = spec.split(',').map(s => s.trim()).filter(s => s.length > 0)
  const tokens: Array<number | Activation> = []
  for (const p of parts) {
    if (ACTIVATIONS.includes(p as Activation)) {
      tokens.push(p as Activation)
    } else if (/^\d+$/.test(p)) {
      const n = parseInt(p, 10)
      if (n <= 0) return { tokens: [], error: `dim must be > 0 (got ${p})` }
      tokens.push(n)
    } else {
      return { tokens: [], error: `unknown spec token '${p}' (expected a positive integer or one of ${ACTIVATIONS.join(',')})` }
    }
  }
  return { tokens }
}

function buildLayers(form: FormState): { layers: Layer[]; inputDim: number; outputDim: number; error?: string } {
  if (form.kind === 'feedforward') {
    const parsed = parseSpec(form.hiddenSpec)
    if (parsed.error) return { layers: [], inputDim: 0, outputDim: 0, error: parsed.error }
    if (form.inputDim <= 0) return { layers: [], inputDim: 0, outputDim: 0, error: 'input dim must be > 0' }
    if (form.outputDim <= 0) return { layers: [], inputDim: 0, outputDim: 0, error: 'output dim must be > 0' }

    const layers: Layer[] = []
    let cur = form.inputDim
    for (const tok of parsed.tokens) {
      if (typeof tok === 'number') {
        layers.push({ type: 'linear', in_dim: cur, out_dim: tok })
        cur = tok
      } else {
        layers.push({ type: 'activation', activation: tok })
      }
    }
    layers.push({ type: 'linear', in_dim: cur, out_dim: form.outputDim })
    if (form.outputActivation !== 'identity') {
      layers.push({ type: 'activation', activation: form.outputActivation })
    }
    return { layers, inputDim: form.inputDim, outputDim: form.outputDim }
  }

  // next_token
  // Reserved tokens (4) + user vocab budget — final vocab built from corpus,
  // but we need to size the input/output layers now. We approximate with the
  // budget; if real vocab differs we'll surface a clear error at training time.
  const vocab = Math.max(form.vocabBudget, 8)
  const inputDim = vocab * form.contextSize
  const parsed = parseSpec(form.embedHidden)
  if (parsed.error) return { layers: [], inputDim: 0, outputDim: 0, error: parsed.error }
  if (form.contextSize <= 0) return { layers: [], inputDim: 0, outputDim: 0, error: 'context size must be > 0' }

  const layers: Layer[] = []
  let cur = inputDim
  for (const tok of parsed.tokens) {
    if (typeof tok === 'number') {
      layers.push({ type: 'linear', in_dim: cur, out_dim: tok })
      cur = tok
    } else {
      layers.push({ type: 'activation', activation: tok })
    }
  }
  // Output layer projects to vocab. Loss is softmax+cross-entropy, so leave logits raw.
  layers.push({ type: 'linear', in_dim: cur, out_dim: vocab })
  return { layers, inputDim, outputDim: vocab }
}

export default function NetworksTab({ refreshNetworks, onSelect }: TabProps & { onSelect: (id: string) => void }) {
  const [list, setList] = useState<Network[]>([])
  const [error, setError] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState<FormState>(defaultForm('feedforward'))
  const [busy, setBusy] = useState(false)

  useEffect(() => { void load() }, [])

  const load = async () => {
    try { setList(await networks.list()); setError(null) }
    catch (e) { setError(String(e)) }
  }

  const preview = useMemo(() => buildLayers(form), [form])

  const onSubmit = async () => {
    if (preview.error) { setError(preview.error); return }
    setBusy(true); setError(null)
    try {
      const created = await networks.create({
        name: form.name.trim() || 'untitled',
        kind: form.kind,
        seed: form.seed,
        layers: preview.layers,
        input_dim: preview.inputDim,
        context_size: form.kind === 'next_token' ? form.contextSize : null,
      })
      setShowForm(false)
      setForm(defaultForm(form.kind))
      await load()
      await refreshNetworks()
      onSelect(created.id)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const onDelete = async (id: string) => {
    try { await networks.delete(id); await load(); await refreshNetworks() }
    catch (e) { setError(String(e)) }
  }

  return (
    <div className="tab-content">
      <h2>Networks</h2>
      <p className="muted">Create and manage neural-network architectures.</p>

      {error && <div className="status error">{error}</div>}

      <div className="card">
        <div className="card-row">
          <h3 style={{ margin: 0 }}>{showForm ? 'Create network' : `${list.length} network${list.length === 1 ? '' : 's'}`}</h3>
          <button onClick={() => setShowForm(s => !s)} className={showForm ? 'secondary' : ''}>
            {showForm ? 'Cancel' : 'New network'}
          </button>
        </div>

        {showForm && (
          <div className="mt-2 flex-col">
            <div className="grid-2">
              <div>
                <label>Name</label>
                <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
              </div>
              <div>
                <label>Type</label>
                <select
                  value={form.kind}
                  onChange={e => {
                    const kind = e.target.value as NetworkKind
                    setForm(defaultForm(kind))
                  }}
                >
                  <option value="feedforward">Feed-forward (regression / classification)</option>
                  <option value="next_token">Next-token prediction (text)</option>
                </select>
              </div>
            </div>

            {form.kind === 'feedforward' ? (
              <>
                <div className="grid-3">
                  <div>
                    <label>Input dim</label>
                    <input type="number" min={1} value={form.inputDim}
                      onChange={e => setForm({ ...form, inputDim: parseInt(e.target.value) || 1 })} />
                  </div>
                  <div>
                    <label>Output dim</label>
                    <input type="number" min={1} value={form.outputDim}
                      onChange={e => setForm({ ...form, outputDim: parseInt(e.target.value) || 1 })} />
                  </div>
                  <div>
                    <label>Output activation</label>
                    <select value={form.outputActivation}
                      onChange={e => setForm({ ...form, outputActivation: e.target.value as Activation })}>
                      {ACTIVATIONS.map(a => <option key={a} value={a}>{a}</option>)}
                    </select>
                  </div>
                </div>
                <div>
                  <label>Hidden layers (comma-separated dims and activations)</label>
                  <input value={form.hiddenSpec}
                    onChange={e => setForm({ ...form, hiddenSpec: e.target.value })}
                    placeholder="8,tanh,4,tanh" />
                  <small>e.g. <code>8,tanh,4,tanh</code> → Linear 2→8 · tanh · Linear 8→4 · tanh · Linear 4→{form.outputDim}</small>
                </div>
              </>
            ) : (
              <>
                <div className="grid-3">
                  <div>
                    <label>Vocab budget</label>
                    <input type="number" min={8} value={form.vocabBudget}
                      onChange={e => setForm({ ...form, vocabBudget: parseInt(e.target.value) || 8 })} />
                    <small>Estimated vocabulary size. Sizes input/output layers. Must match real corpus vocab at train time.</small>
                  </div>
                  <div>
                    <label>Context size</label>
                    <input type="number" min={1} max={512} value={form.contextSize}
                      onChange={e => setForm({ ...form, contextSize: parseInt(e.target.value) || 1 })} />
                    <small>Tokens fed per prediction.</small>
                  </div>
                  <div>
                    <label>Seed</label>
                    <input type="number" value={form.seed}
                      onChange={e => setForm({ ...form, seed: parseInt(e.target.value) || 0 })} />
                  </div>
                </div>
                <div>
                  <label>Hidden layers (between input and vocab projection)</label>
                  <input value={form.embedHidden}
                    onChange={e => setForm({ ...form, embedHidden: e.target.value })}
                    placeholder="64,relu,32,relu" />
                  <small>Output layer projects to <code>vocab_size</code> logits (softmax+cross-entropy is applied at training time).</small>
                </div>
              </>
            )}

            {form.kind === 'feedforward' && (
              <div>
                <label>Seed</label>
                <input type="number" value={form.seed}
                  onChange={e => setForm({ ...form, seed: parseInt(e.target.value) || 0 })} />
              </div>
            )}

            <ArchitecturePreview preview={preview} />

            <div className="flex">
              <button onClick={onSubmit} disabled={busy || !!preview.error}>
                {busy ? 'Creating…' : 'Create network'}
              </button>
              <button className="secondary" onClick={() => setShowForm(false)}>Cancel</button>
            </div>
          </div>
        )}
      </div>

      {list.length === 0 ? (
        <div className="card"><p className="muted">No networks yet.</p></div>
      ) : list.map(n => (
        <div key={n.id} className="list-item">
          <div className="flex-1">
            <strong>{n.name}</strong>
            <p>
              <span className="chip">{n.kind === 'next_token' ? 'next-token' : 'feed-forward'}</span>
              {' '}<span className="chip">{n.input_dim} → {n.output_dim}</span>
              {' '}<span className="chip">{n.layers.length} layers</span>
              {' '}<span className="chip">{n.parameter_count.toLocaleString()} params</span>
              {n.context_size ? <> {' '}<span className="chip">ctx {n.context_size}</span></> : null}
              {n.trained ? <> {' '}<span className="chip accent">trained</span></> : null}
            </p>
          </div>
          <div className="flex">
            <ExportButton network={n} onError={setError} />
            <button className="danger" onClick={() => onDelete(n.id)}>Delete</button>
          </div>
        </div>
      ))}
    </div>
  )
}

function ExportButton({ network, onError }: { network: Network; onError: (e: string) => void }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<ExportFormat | null>(null)

  const formats: { id: ExportFormat; label: string; enabled: boolean; reason?: string }[] = [
    { id: 'pytorch', label: 'PyTorch (.pt)', enabled: true },
    { id: 'onnx',    label: 'ONNX (.onnx)', enabled: true },
    {
      id: 'gguf', label: 'GGUF (.gguf)',
      enabled: network.kind === 'next_token',
      reason: network.kind === 'next_token' ? undefined : 'GGUF is only useful for next-token networks',
    },
  ]

  const doExport = async (fmt: ExportFormat) => {
    setBusy(fmt)
    try {
      const payload = await exporter.run(network.id, fmt)
      // Decode base64 → Blob → trigger download.
      const bin = atob(payload.data_b64)
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
      const blob = new Blob([bytes], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = payload.filename
      document.body.appendChild(a); a.click()
      document.body.removeChild(a); URL.revokeObjectURL(url)
      setOpen(false)
    } catch (e) {
      onError(String(e))
    } finally { setBusy(null) }
  }

  if (!open) {
    return <button className="secondary" onClick={() => setOpen(true)} disabled={!network.trained && network.kind === 'next_token'}>
      Export
    </button>
  }

  return (
    <div className="flex" style={{ gap: 4 }}>
      {formats.map(f => (
        <button
          key={f.id}
          className="secondary"
          disabled={!f.enabled || busy !== null}
          title={f.reason}
          onClick={() => doExport(f.id)}
        >
          {busy === f.id ? 'Exporting…' : f.label}
        </button>
      ))}
      <button className="secondary" onClick={() => setOpen(false)}>Cancel</button>
    </div>
  )
}

function ArchitecturePreview({ preview }: {
  preview: { layers: Layer[]; inputDim: number; outputDim: number; error?: string }
}) {
  if (preview.error) {
    return <div className="status error">{preview.error}</div>
  }
  return (
    <div className="card" style={{ background: 'var(--bg-input)', margin: 0 }}>
      <h4>Architecture preview</h4>
      <div className="flex" style={{ flexWrap: 'wrap' }}>
        <span className="chip">in {preview.inputDim}</span>
        {preview.layers.map((l, i) => (
          <span key={i} className="chip">
            {l.type === 'linear' ? `linear → ${l.out_dim}` : l.activation}
          </span>
        ))}
        <span className="chip accent">out {preview.outputDim}</span>
      </div>
    </div>
  )
}
