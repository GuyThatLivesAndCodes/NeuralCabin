import { useEffect, useMemo, useState } from 'react'
import { vocabulary, VocabularyInfo } from '../api'
import type { TabProps } from '../App'

const RESERVED = ['<pad>', '<unk>', '<bos>', '<eos>']

export default function VocabTab({ network }: TabProps) {
  const [vocab, setVocab] = useState<VocabularyInfo | null>(null)
  const [filter, setFilter] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (network) void load(network.id)
    else setVocab(null)
  }, [network])

  const load = async (id: string) => {
    setError(null)
    try {
      const v = await vocabulary.get(id)
      setVocab(v)
    } catch (e) { setError(String(e)); setVocab(null) }
  }

  const filtered = useMemo(() => {
    if (!vocab?.tokens) return []
    const tokens = vocab.tokens
    if (!filter.trim()) return tokens.map((t, i) => ({ id: i, token: t }))
    const q = filter.toLowerCase()
    return tokens
      .map((t, i) => ({ id: i, token: t }))
      .filter(({ token }) => token.toLowerCase().includes(q))
  }, [vocab?.tokens, filter])

  const onExport = () => {
    if (!vocab) return
    const data = { mode: vocab.mode, tokens: vocab.tokens }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `${network?.name ?? 'vocab'}-vocabulary.json`
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

      {!network ? (
        <div className="card"><p className="muted">Select a network.</p></div>
      ) : network.kind !== 'next_token' ? (
        <div className="card">
          <p className="muted">
            <strong>{network.name}</strong> is a feed-forward network. Vocabularies
            apply only to next-token networks.
          </p>
        </div>
      ) : !vocab?.tokens ? (
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
                <h3 style={{ margin: 0 }}>{vocab.tokens.length.toLocaleString()} tokens</h3>
                <p className="muted" style={{ marginTop: 4 }}>
                  Mode: <span className="chip">{vocab.mode ?? 'unknown'}</span>
                  {' '}Reserved: <span className="chip">{RESERVED.length}</span>
                  {' '}User: <span className="chip">{vocab.tokens.length - RESERVED.length}</span>
                  {network.context_size ? <> · Context: <span className="chip">{network.context_size}</span></> : null}
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
