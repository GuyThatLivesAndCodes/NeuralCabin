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

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'docs', label: 'Documentation', icon: '📖' },
    { id: 'networks', label: 'Networks', icon: '🧠' },
    { id: 'corpus', label: 'Corpus', icon: '📚' },
    { id: 'vocab', label: 'Vocabulary', icon: '📝' },
    { id: 'training', label: 'Training', icon: '⚡' },
    { id: 'inference', label: 'Inference', icon: '🔮' },
    { id: 'plugins', label: 'Plugins', icon: '🔌' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header>
        <div style={{ padding: '20px 24px' }}>
          <h1>🧠 NeuralCabin</h1>
          <p style={{ margin: '4px 0 0 0', color: '#9b8a7f', fontSize: '14px' }}>
            Pure Rust Neural Network Workbench
          </p>
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <nav className="tabs" style={{ width: '200px' }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
              style={{
                fontSize: '14px',
                paddingLeft: '16px',
              }}
            >
              <span style={{ marginRight: '8px' }}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        <div style={{ flex: 1, overflow: 'auto', background: 'linear-gradient(135deg, #faf8f3 0%, #f5ede2 100%)' }}>
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
