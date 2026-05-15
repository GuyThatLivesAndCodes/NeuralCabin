import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'

// ─── Network ────────────────────────────────────────────────────────────────

export type NetworkKind = 'feedforward' | 'next_token'

export type Activation = 'identity' | 'relu' | 'sigmoid' | 'tanh' | 'softmax'

export type Layer =
  | { type: 'linear'; in_dim: number; out_dim: number }
  | { type: 'activation'; activation: Activation }

export interface Network {
  id: string
  name: string
  kind: NetworkKind
  layers: Layer[]
  seed: number
  input_dim: number
  output_dim: number
  created_at: string
  parameter_count: number
  trained: boolean
  context_size?: number | null
}

export interface CreateNetworkRequest {
  name: string
  kind: NetworkKind
  seed: number
  layers: Layer[]
  input_dim: number
  context_size?: number | null
}

export const networks = {
  create: (req: CreateNetworkRequest) => invoke<Network>('create_network', { req }),
  list:   ()                            => invoke<Network[]>('list_networks'),
  get:    (id: string)                  => invoke<Network>('get_network', { id }),
  delete: (id: string)                  => invoke<boolean>('delete_network', { id }),
}

// ─── Corpus ─────────────────────────────────────────────────────────────────

export interface FeedforwardCorpus {
  features: number[]
  targets: number[]
  rows: number
  in_dim: number
  out_dim: number
}

export interface FineTunePair {
  input: string
  output: string
}

export type GptStage = 'pretrain' | 'finetune'

export interface SetCorpusRequest {
  network_id: string
  feedforward?: FeedforwardCorpus
  text?: string
  pairs?: FineTunePair[]
  vocab_mode?: 'char' | 'word'
  stage?: GptStage
}

export interface Corpus {
  network_id: string
  kind: NetworkKind
  updated_at: string
  feedforward?: FeedforwardCorpus
  text?: string
  pairs?: FineTunePair[]
  vocab_mode?: string
  vocab?: string[]
  stage?: GptStage
}

export interface CorpusStats {
  kind: NetworkKind
  stage?: GptStage
  rows?: number
  in_dim?: number
  out_dim?: number
  text_chars?: number
  text_tokens?: number
  pair_count?: number
  vocab_size?: number
  vocab_mode?: string
  training_examples?: number
}

export const corpus = {
  set:   (req: SetCorpusRequest)        => invoke<CorpusStats>('set_corpus', { req }),
  get:   (network_id: string)           => invoke<Corpus | null>('get_corpus', { networkId: network_id }),
  stats: (network_id: string)           => invoke<CorpusStats | null>('corpus_stats', { networkId: network_id }),
}

// ─── Vocabulary ─────────────────────────────────────────────────────────────

export const vocabulary = {
  get: (network_id: string) =>
    invoke<string[] | null>('get_vocabulary', { networkId: network_id }),
}

// ─── Training ───────────────────────────────────────────────────────────────

export interface OptimizerConfig {
  kind: 'adam' | 'sgd'
  lr: number
  beta1?: number
  beta2?: number
  eps?: number
  momentum?: number
}

export interface TrainingConfig {
  epochs: number
  batch_size: number
  optimizer: OptimizerConfig
  loss: 'mse' | 'crossentropy'
  seed: number
  mask_user_tokens?: boolean
}

export interface TrainingRequest {
  network_id: string
  config: TrainingConfig
}

export interface TrainingStatus {
  training_id: string
  status: 'running' | 'completed' | 'error'
  epoch: number
  total_epochs: number
  last_loss: number
  loss_history: number[]
  elapsed_secs: number
}

export interface TrainingUpdate {
  training_id: string
  epoch: number
  total_epochs: number
  loss: number
  loss_history: number[]
  elapsed_secs: number
}

export interface TrainingFinished {
  training_id: string
  status: string
  final_loss: number
  total_epochs: number
  elapsed_secs: number
}

export interface TrainingError {
  training_id: string
  message: string
}

export const training = {
  start:  (req: TrainingRequest) =>
    invoke<{ training_id: string; status: string }>('start_training', { req }),

  status: (trainingId: string) =>
    invoke<TrainingStatus>('get_training_status', { trainingId }),

  onUpdate: (handler: (u: TrainingUpdate) => void): Promise<UnlistenFn> =>
    listen<TrainingUpdate>('training_update', (e) => handler(e.payload)),

  onFinished: (handler: (r: TrainingFinished) => void): Promise<UnlistenFn> =>
    listen<TrainingFinished>('training_finished', (e) => handler(e.payload)),

  onError: (handler: (err: TrainingError) => void): Promise<UnlistenFn> =>
    listen<TrainingError>('training_error', (e) => handler(e.payload)),
}

// ─── Inference ──────────────────────────────────────────────────────────────

export interface InferRequest {
  network_id: string
  features?: number[]
  prompt?: string
  max_new_tokens?: number
  temperature?: number
}

export interface GenerationStep {
  token: string
  probability: number
}

export interface InferResponse {
  output?: number[]
  generated?: string
  steps?: GenerationStep[]
}

export const inference = {
  run: (req: InferRequest) => invoke<InferResponse>('infer', { req }),
}
