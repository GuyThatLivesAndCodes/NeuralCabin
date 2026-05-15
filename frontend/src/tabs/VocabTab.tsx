import { useEffect, useMemo, useState } from 'react'
import { networks, vocabulary, corpus, Network } from '../api'

const RESERVED = ['<pad>', '<unk>', '<bos>', '<eos>']

export default function VocabTab() {
  const [list, setList] = useState<Network[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [tokens, setTokens] = useState<string[] | null>(null)
  const [vocabMode, setVocabMode] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { void loadNetworks() }, [])
  useEffect(() => { if (selectedId) void load(selectedId) }, [selectedId])

  const loadNetworks = async () => {
    try {
      const items = await networks.list()
      setList(items)
      // Auto-select first next-token network if any
      const txt = items.find(n => n.kind === 'next_token')
      if (txt) setSelectedId(txt.id)
      else if (items.length) setSelectedId(items[0].id)
    } catch (e) { setError(String(e)) }
  }

  const load = async (id: string) => {
    setError(null)
    try {
      const [v, c] = await Promise.all([vocabulary.get(id), corpus.get(id)])
      setTokens(v)
      setVocabMode(c?.vocab_mode ?? null)
    } catch (e) { setError(String(e)); setTokens(null) }
  }

  const selected = list.find(n => n.id === selectedId)

  const filtered = useMemo(() => {
    if (!tokens) return []
    if (!filter.trim()) return tokens.map((t, i) => ({ id: i, token: t }))
    const q = filter.toLowerCase()
    return tokens
      .map((t, i) => ({ id: i, token: t }))
      .filter(({ token }) => token.toLowerCase().includes(q))
  }, [tokens, filter])

  const onExport = () => {
    if (!tokens) return
    const data = { mode: vocabMode, tokens }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `${selected?.name ?? 'vocab'}-vocabulary.json`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="tab-content">
      <h2>Vocabulary</h2>
      <p className="muted">
        Vocabularies are built automatically from the corpus of next-token
        networks. Reserved tokens (<code>{RESERVED.join(' ')}</code>) always occupy
        the first four indices.
      </p>

      {error && <div className="status error">{error}</div>}

      <div className="card">
        <label>Network</label>
        <select value={selectedId} onChange={e => setSelectedId(e.target.value)}>
          <option value="">— Select a network —</option>
          {list.map(n => (
            <option key={n.id} value={n.id} disabled={n.kind !== 'next_token'}>
              {n.name} {n.kind !== 'next_token' ? '(no vocabulary — only next-token networks have one)' : ''}
            </option>
          ))}
        </select>
      </div>

      {!selected ? (
        <div className="card"><p className="muted">Select a network.</p></div>
      ) : selected.kind !== 'next_token' ? (
        <div className="card">
          <p className="muted">
            <strong>{selected.name}</strong> is a feed-forward network. Vocabularies
            apply only to next-token networks.
          </p>
        </div>
      ) : !tokens ? (
        <div className="card">
          <p className="muted">
            No vocabulary yet. Set a corpus on the <strong>Corpus</strong> tab
            and the vocabulary will be built automatically.
          </p>
        </div>
      ) : (
        <>
          <div className="card">
            <div className="card-row">
              <div>
                <h3 style={{ margin: 0 }}>{tokens.length.toLocaleString()} tokens</h3>
                <p className="muted" style={{ marginTop: 4 }}>
                  Mode: <span className="chip">{vocabMode ?? 'unknown'}</span>
                  {' '}Reserved: <span className="chip">{RESERVED.length}</span>
                  {' '}User: <span className="chip">{tokens.length - RESERVED.length}</span>
                  {selected.context_size ? <> · Context: <span className="chip">{selected.context_size}</span></> : null}
                </p>
              </div>
              <button className="secondary" onClick={onExport}>Export JSON</button>
            </div>
          </div>

          <div className="card">
            <input
              placeholder="Filter tokens…"
              value={filter}
              onChange={e => setFilter(e.target.value)}
            />
            <div style={{ maxHeight: 480, overflow: 'auto', marginTop: 12 }}>
              <table>
                <thead>
                  <tr><th style={{ width: 80 }}>ID</th><th>Token</th></tr>
                </thead>
                <tbody>
                  {filtered.map(({ id, token }) => (
                    <tr key={id}>
                      <td><code>{id}</code></td>
                      <td>
                        {id < RESERVED.length
                          ? <span className="token-special">{token}</span>
                          : <code>{token}</code>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filtered.length === 0 && (
                <p className="muted" style={{ padding: 16, textAlign: 'center' }}>No matches.</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
