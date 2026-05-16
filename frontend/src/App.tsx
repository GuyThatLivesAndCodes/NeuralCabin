import { useCallback, useEffect, useState } from 'react'
import NetworksTab from './tabs/NetworksTab'
import CorpusTab from './tabs/CorpusTab'
import VocabTab from './tabs/VocabTab'
import TrainingTab from './tabs/TrainingTab'
import InferenceTab from './tabs/InferenceTab'
import DocsTab from './tabs/DocsTab'
import ServerTab from './tabs/ServerTab'
import SettingsTab from './tabs/SettingsTab'
import { networks, Network } from './api'
import { applySettings, loadSettings } from './settings'

type Tab = 'networks' | 'corpus' | 'vocab' | 'training' | 'inference' | 'docs' | 'server' | 'settings'

const TABS: { id: Tab; label: string }[] = [
  { id: 'networks',  label: 'Networks' },
  { id: 'corpus',    label: 'Corpus' },
  { id: 'vocab',     label: 'Vocabulary' },
  { id: 'training',  label: 'Training' },
  { id: 'inference', label: 'Inference' },
  { id: 'docs',      label: 'Documentation' },
  { id: 'server',    label: 'Server' },
  { id: 'settings',  label: 'Settings' },
]

export interface TabProps {
  network: Network | null
  refreshNetworks: () => Promise<void>
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('networks')
  const [list, setList] = useState<Network[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const items = await networks.list()
      setList(items)
      // If our selection has gone away (e.g. user deleted), reset.
      setSelectedId(prev => {
        if (prev && items.some(n => n.id === prev)) return prev
        return items.length > 0 ? items[0].id : null
      })
    } catch {
      setList([])
      setSelectedId(null)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  // Apply user theme settings on boot so the primary color picked in
  // Settings persists across restarts.
  useEffect(() => { applySettings(loadSettings()) }, [])

  const selected = list.find(n => n.id === selectedId) ?? null
  const props: TabProps = { network: selected, refreshNetworks: refresh }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header className="app-header">
        <h1>NeuralCabin</h1>
        <span className="subtitle">Pure Rust neural-network workbench</span>
        <div className="spacer" />
        <div className="network-picker">
          <label className="picker-label">Network</label>
          <select
            value={selectedId ?? ''}
            onChange={e => setSelectedId(e.target.value || null)}
            disabled={list.length === 0}
          >
            {list.length === 0 ? (
              <option value="">— no networks yet —</option>
            ) : (
              list.map(n => (
                <option key={n.id} value={n.id}>
                  {n.name} · {n.kind === 'next_token' ? 'next-token' : 'feed-forward'}
                  {n.trained ? ' · trained' : ''}
                </option>
              ))
            )}
          </select>
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <main style={{ flex: 1, overflow: 'auto' }}>
          {activeTab === 'networks'  && <NetworksTab {...props} onSelect={setSelectedId} />}
          {activeTab === 'corpus'    && <CorpusTab    {...props} />}
          {activeTab === 'vocab'     && <VocabTab     {...props} />}
          {activeTab === 'training'  && <TrainingTab  {...props} />}
          {activeTab === 'inference' && <InferenceTab {...props} />}
          {activeTab === 'docs'      && <DocsTab      networks={list} />}
          {activeTab === 'server'    && <ServerTab    networks={list} />}
          {activeTab === 'settings'  && <SettingsTab  onChange={() => { /* re-render via state in tab */ }} />}
        </main>
      </div>
    </div>
  )
}
