import { useEffect, useState } from 'react'
import {
  servers, Network,
  ServerSummary, ServerPermissions, CreateServerRequest, UpdateServerRequest,
} from '../api'

const DEFAULT_PERMS: ServerPermissions = {
  allow_list: true,
  allow_inference: true,
  allow_export: true,
  allow_upload: false,
  allow_train: false,
  allow_create: false,
  allow_delete: false,
}

interface FormState {
  name: string
  port: number
  localhost_only: boolean
  auth_token: string
  permissions: ServerPermissions
}

const DEFAULT_FORM: FormState = {
  name: 'local-api',
  port: 8787,
  localhost_only: true,
  auth_token: '',
  permissions: { ...DEFAULT_PERMS },
}

export default function ServerTab({ networks: nets }: { networks: Network[] }) {
  const [list, setList] = useState<ServerSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState<FormState>(DEFAULT_FORM)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  const refresh = async () => {
    try { setList(await servers.list()); setError(null) }
    catch (e) { setError(String(e)) }
  }
  useEffect(() => { void refresh() }, [])
  // Periodic refresh while any server is running, so request_count updates.
  useEffect(() => {
    const hasRunning = list.some(s => s.running)
    if (!hasRunning) return
    const id = window.setInterval(() => setTick(t => t + 1), 2000)
    return () => window.clearInterval(id)
  }, [list])
  useEffect(() => { if (tick > 0) void refresh() }, [tick])

  const onCreate = async () => {
    setBusy(true); setError(null)
    try {
      const req: CreateServerRequest = {
        name: form.name.trim() || 'local-api',
        port: form.port,
        localhost_only: form.localhost_only,
        auth_token: form.auth_token,
        permissions: form.permissions,
      }
      await servers.create(req)
      setShowCreate(false)
      setForm(DEFAULT_FORM)
      await refresh()
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  const onSave = async (id: string) => {
    setBusy(true); setError(null)
    try {
      const req: UpdateServerRequest = {
        id,
        name: form.name,
        port: form.port,
        localhost_only: form.localhost_only,
        auth_token: form.auth_token,
        permissions: form.permissions,
      }
      await servers.update(req)
      setEditingId(null)
      await refresh()
    } catch (e) { setError(String(e)) }
    finally { setBusy(false) }
  }

  const onStart  = async (id: string) => { try { await servers.start(id); await refresh() } catch (e) { setError(String(e)) } }
  const onStop   = async (id: string) => { try { await servers.stop(id);  await refresh() } catch (e) { setError(String(e)) } }
  const onDelete = async (id: string) => {
    if (!confirm('Delete this server? It will be stopped first.')) return
    try { await servers.delete(id); await refresh() } catch (e) { setError(String(e)) }
  }

  const beginEdit = (s: ServerSummary) => {
    setEditingId(s.id)
    setForm({
      name: s.name, port: s.port,
      localhost_only: s.localhost_only,
      auth_token: s.auth_token,
      permissions: { ...s.permissions },
    })
    setShowCreate(false)
  }

  return (
    <div className="tab-content">
      <h2>Server</h2>
      <p className="muted">
        Run HTTP servers on your machine that expose the workbench to other
        users on your network. Remote users can list networks, run inference,
        upload weights, trigger training, and download trained models — only
        for the operations you enable per server. You stay in control of the
        port, the bearer token, and which permissions are granted.
      </p>

      {error && <div className="status error">{error}</div>}

      <div className="card">
        <div className="card-row">
          <h3 style={{ margin: 0 }}>
            {showCreate ? 'New server' : `${list.length} server${list.length === 1 ? '' : 's'}`}
          </h3>
          <button
            onClick={() => { setShowCreate(s => !s); setEditingId(null); setForm(DEFAULT_FORM) }}
            className={showCreate ? 'secondary' : ''}
          >
            {showCreate ? 'Cancel' : 'New server'}
          </button>
        </div>

        {showCreate && (
          <ServerForm form={form} setForm={setForm} onSubmit={onCreate} busy={busy} submitLabel="Create server" />
        )}
      </div>

      {list.length === 0 ? (
        <div className="card"><p className="muted">No servers yet. Create one above to expose your workbench over HTTP.</p></div>
      ) : list.map(s => (
        <div key={s.id} className="card">
          <div className="card-row">
            <div className="flex-1">
              <strong>{s.name}</strong>
              <p>
                <span className="chip">{s.localhost_only ? 'localhost' : 'public'}:{s.port}</span>
                {' '}<span className={`chip ${s.running ? 'accent' : ''}`}>{s.running ? 'running' : 'stopped'}</span>
                {s.auth_token ? <> {' '}<span className="chip">auth: bearer</span></> : <> {' '}<span className="chip">no auth</span></>}
                {s.auto_start ? <> {' '}<span className="chip">auto-start</span></> : null}
                {s.running ? <> {' '}<span className="chip">{s.request_count} req</span></> : null}
              </p>
              {s.last_error && <div className="status error">{s.last_error}</div>}
              {s.running && (
                <p className="muted" style={{ marginTop: 4 }}>
                  Open in browser:{' '}
                  <a
                    href={`http://${s.localhost_only ? '127.0.0.1' : 'localhost'}:${s.port}/`}
                    target="_blank" rel="noreferrer"
                  >
                    http://{s.localhost_only ? '127.0.0.1' : 'localhost'}:{s.port}/
                  </a>
                </p>
              )}
            </div>
            <div className="flex">
              {!s.running
                ? <button onClick={() => onStart(s.id)}>Start</button>
                : <button className="secondary" onClick={() => onStop(s.id)}>Stop</button>}
              {!s.running && editingId !== s.id && (
                <button className="secondary" onClick={() => beginEdit(s)}>Edit</button>
              )}
              <button className="danger" onClick={() => onDelete(s.id)}>Delete</button>
            </div>
          </div>

          {editingId === s.id && (
            <ServerForm
              form={form} setForm={setForm}
              onSubmit={() => onSave(s.id)} busy={busy} submitLabel="Save changes"
              onCancel={() => setEditingId(null)}
            />
          )}

          {s.running && (
            <UsageDocs server={s} networks={nets} />
          )}
        </div>
      ))}

      <div className="card">
        <h3>What this is for</h3>
        <p>
          Each server is a small HTTP API hosted from inside NeuralCabin. The
          most common workflow: another person on your network (or you, from a
          script) uploads neural network weights, triggers training using your
          GPU, polls progress, and downloads the trained model.
        </p>
        <p>
          Permissions are off-by-default for write actions. <code>inference</code>,
          <code>list</code>, and <code>export</code> are on by default, since they
          don't change anything on your machine. Turn on <code>upload</code>,
          <code>train</code>, <code>create</code>, or <code>delete</code> explicitly
          per server.
        </p>
        <p>
          When auth is configured, every request must carry
          <code> Authorization: Bearer &lt;token&gt;</code>. Without a token, anyone
          who can reach the port can hit the API — keep <em>localhost only</em>
          on unless you specifically want LAN access.
        </p>
      </div>
    </div>
  )
}

function ServerForm({ form, setForm, onSubmit, busy, submitLabel, onCancel }: {
  form: FormState
  setForm: (f: FormState) => void
  onSubmit: () => void
  busy: boolean
  submitLabel: string
  onCancel?: () => void
}) {
  const togglePerm = (k: keyof ServerPermissions) =>
    setForm({ ...form, permissions: { ...form.permissions, [k]: !form.permissions[k] } })

  return (
    <div className="mt-2 flex-col">
      <div className="grid-3">
        <div>
          <label>Name</label>
          <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
        </div>
        <div>
          <label>Port</label>
          <input type="number" min={1} max={65535} value={form.port}
            onChange={e => setForm({ ...form, port: parseInt(e.target.value) || 0 })} />
        </div>
        <div>
          <label>Bind</label>
          <select value={form.localhost_only ? '1' : '0'}
            onChange={e => setForm({ ...form, localhost_only: e.target.value === '1' })}>
            <option value="1">localhost only (127.0.0.1)</option>
            <option value="0">all interfaces (0.0.0.0)</option>
          </select>
        </div>
      </div>
      <div>
        <label>Bearer token (empty = no auth required)</label>
        <input value={form.auth_token} onChange={e => setForm({ ...form, auth_token: e.target.value })}
          placeholder="e.g. 9f4c7e... — paste or invent any opaque string" />
      </div>
      <div>
        <label>Permissions</label>
        <div className="flex" style={{ flexWrap: 'wrap', gap: 8 }}>
          {(['allow_list','allow_inference','allow_export','allow_upload','allow_train','allow_create','allow_delete'] as (keyof ServerPermissions)[]).map(k => (
            <label key={k} className="flex" style={{ gap: 4 }}>
              <input type="checkbox" checked={form.permissions[k]} onChange={() => togglePerm(k)} />
              {k.replace('allow_', '')}
            </label>
          ))}
        </div>
        <small className="muted">
          <code>list</code>/<code>inference</code>/<code>export</code> are read-only.
          {' '}<code>upload</code> overwrites weights, <code>train</code> uses your GPU,
          {' '}<code>create</code>/<code>delete</code> manage networks.
        </small>
      </div>
      <div className="flex">
        <button onClick={onSubmit} disabled={busy}>{busy ? 'Working…' : submitLabel}</button>
        {onCancel && <button className="secondary" onClick={onCancel}>Cancel</button>}
      </div>
    </div>
  )
}

function UsageDocs({ server, networks }: { server: ServerSummary; networks: Network[] }) {
  const host = server.localhost_only ? '127.0.0.1' : 'localhost'
  const base = `http://${host}:${server.port}`
  const authHdr = server.auth_token ? `-H 'Authorization: Bearer ${server.auth_token}' ` : ''
  const ff = networks.find(n => n.kind === 'feedforward')
  const nt = networks.find(n => n.kind === 'next_token')
  const exampleId = ff?.id ?? nt?.id ?? '<NETWORK_ID>'

  return (
    <details style={{ marginTop: 12 }}>
      <summary><strong>How to use this server</strong></summary>
      <div className="mt-2">
        <p className="muted">
          The full reference is available at <a href={base + '/'} target="_blank" rel="noreferrer">{base}/</a>.
          Quick examples below.
        </p>

        <p><strong>List networks</strong> (open in browser too):</p>
        <pre>{`curl ${authHdr}${base}/api/networks`}</pre>

        {server.permissions.allow_inference && ff && (
          <>
            <p><strong>Run feed-forward inference</strong>:</p>
            <pre>{`curl -X POST ${authHdr}${base}/api/networks/${ff.id}/infer \\
  -H 'Content-Type: application/json' \\
  -d '{"features":[${Array(ff.input_dim).fill(0).join(',')}]}'`}</pre>
          </>
        )}

        {server.permissions.allow_inference && nt && (
          <>
            <p><strong>Generate next-token text</strong>:</p>
            <pre>{`curl -X POST ${authHdr}${base}/api/networks/${nt.id}/infer \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"hello","max_new_tokens":32,"temperature":0.7}'`}</pre>
          </>
        )}

        {server.permissions.allow_export && (
          <>
            <p><strong>Export to PyTorch / ONNX / GGUF</strong>:</p>
            <pre>{`curl ${authHdr}'${base}/api/networks/${exampleId}/export?format=onnx' -o model.onnx`}</pre>
          </>
        )}

        {server.permissions.allow_upload && (
          <>
            <p><strong>Upload weights</strong> (JSON dump of NeuralCabin's model format — download via <code>/weights</code> first to see the schema):</p>
            <pre>{`curl -X POST ${authHdr}${base}/api/networks/${exampleId}/weights \\
  -H 'Content-Type: application/json' --data-binary @weights.json`}</pre>
          </>
        )}

        {server.permissions.allow_train && (
          <>
            <p><strong>Kick training</strong>:</p>
            <pre>{`curl -X POST ${authHdr}${base}/api/networks/${exampleId}/train \\
  -H 'Content-Type: application/json' \\
  -d '{"epochs":20,"batch_size":16,"loss":"mse","seed":1,
       "optimizer":{"kind":"adam","lr":0.01}}'

# returns {"training_id":"..."}; then poll:
curl ${authHdr}${base}/api/training/<TRAINING_ID>`}</pre>
          </>
        )}

        {server.permissions.allow_create && (
          <>
            <p><strong>Create a network remotely</strong>:</p>
            <pre>{`curl -X POST ${authHdr}${base}/api/networks \\
  -H 'Content-Type: application/json' \\
  -d '{"name":"remote-mlp","kind":"feedforward","seed":1,
       "input_dim":2,
       "layers":[
         {"type":"linear","in_dim":2,"out_dim":8},
         {"type":"activation","activation":"relu"},
         {"type":"linear","in_dim":8,"out_dim":1}
       ]}'`}</pre>
          </>
        )}
      </div>
    </details>
  )
}
