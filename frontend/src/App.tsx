import { useState } from 'react'
import NetworksTab from './tabs/NetworksTab'
import CorpusTab from './tabs/CorpusTab'
import VocabTab from './tabs/VocabTab'
import TrainingTab from './tabs/TrainingTab'
import InferenceTab from './tabs/InferenceTab'
import PluginsTab from './tabs/PluginsTab'
import DocsTab from './tabs/DocsTab'

type Tab = 'docs' | 'networks' | 'corpus' | 'vocab' | 'training' | 'inference' | 'plugins'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('networks')

  const tabs: { id: Tab; label: string }[] = [
    { id: 'docs', label: 'Docs' },
    { id: 'networks', label: 'Networks' },
    { id: 'corpus', label: 'Corpus' },
    { id: 'vocab', label: 'Vocab' },
    { id: 'training', label: 'Training' },
    { id: 'inference', label: 'Inference' },
    { id: 'plugins', label: 'Plugins' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{ padding: '16px', background: '#1a1a1a', borderBottom: '1px solid #404040' }}>
        <h1 style={{ margin: 0 }}>NeuralCabin</h1>
      </header>

      <div style={{ display: 'flex', flex: 1 }}>
        <nav className="tabs" style={{ flexDirection: 'column', width: 'auto', borderBottom: 'none', borderRight: '1px solid #404040' }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
              style={{
                borderBottom: 'none',
                borderRight: activeTab === tab.id ? '4px solid #0066cc' : '4px solid transparent'
              }}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div style={{ flex: 1, overflow: 'auto' }}>
          {activeTab === 'docs' && <DocsTab />}
          {activeTab === 'networks' && <NetworksTab />}
          {activeTab === 'corpus' && <CorpusTab />}
          {activeTab === 'vocab' && <VocabTab />}
          {activeTab === 'training' && <TrainingTab />}
          {activeTab === 'inference' && <InferenceTab />}
          {activeTab === 'plugins' && <PluginsTab />}
        </div>
      </div>
    </div>
  )
}

export default App
