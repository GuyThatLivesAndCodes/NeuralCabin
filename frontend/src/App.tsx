import { useState } from 'react'
import NetworksTab from './tabs/NetworksTab'
import CorpusTab from './tabs/CorpusTab'
import VocabTab from './tabs/VocabTab'
import TrainingTab from './tabs/TrainingTab'
import InferenceTab from './tabs/InferenceTab'
import DocsTab from './tabs/DocsTab'

type Tab = 'networks' | 'corpus' | 'vocab' | 'training' | 'inference' | 'docs'

const TABS: { id: Tab; label: string }[] = [
  { id: 'networks',  label: 'Networks' },
  { id: 'corpus',    label: 'Corpus' },
  { id: 'vocab',     label: 'Vocabulary' },
  { id: 'training',  label: 'Training' },
  { id: 'inference', label: 'Inference' },
  { id: 'docs',      label: 'Documentation' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('networks')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header className="app-header">
        <h1>NeuralCabin</h1>
        <span className="subtitle">Pure Rust neural-network workbench</span>
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
          {activeTab === 'networks'  && <NetworksTab />}
          {activeTab === 'corpus'    && <CorpusTab />}
          {activeTab === 'vocab'     && <VocabTab />}
          {activeTab === 'training'  && <TrainingTab />}
          {activeTab === 'inference' && <InferenceTab />}
          {activeTab === 'docs'      && <DocsTab />}
        </main>
      </div>
    </div>
  )
}
