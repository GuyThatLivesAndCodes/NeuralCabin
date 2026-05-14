import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'

export interface Network {
  id: string
  name: string
  kind: string
  layers: Layer[]
  seed: number
  created_at: string
}

export interface Layer {
  type: 'linear' | 'activation'
  in_dim?: number
  out_dim?: number
  activation?: string
}

export interface Dataset {
  id: string
  name: string
  kind: string
  samples: number
  features: number
  labels: number
  task: string
  created_at: string
}

export interface OptimizerConfig {
  kind: string
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
  loss: string
  validation_frac: number
  seed: number
}

export interface TrainingRequest {
  network_id: string
  dataset_id: string
  config: TrainingConfig
}

export interface TrainingStatus {
  training_id: string
  status: string
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

export const networks = {
  create: (req: { name: string; kind: string; seed: number; layers: Layer[] }) =>
    invoke<Network>('create_network', { req }),

  list: () => invoke<Network[]>('list_networks'),

  get: (id: string) => invoke<Network>('get_network', { id }),

  delete: (id: string) => invoke<boolean>('delete_network', { id }),
}

export const datasets = {
  create: (req: { name: string; kind: string; seed: number }) =>
    invoke<Dataset>('create_dataset', { req }),

  list: () => invoke<Dataset[]>('list_datasets'),
}

export const training = {
  start: (req: TrainingRequest) =>
    invoke<{ training_id: string; status: string }>('start_training', { req }),

  status: (trainingId: string) =>
    invoke<TrainingStatus>('get_training_status', { training_id: trainingId }),

  onUpdate: (handler: (u: TrainingUpdate) => void): Promise<UnlistenFn> =>
    listen<TrainingUpdate>('training_update', (e) => handler(e.payload)),

  onFinished: (handler: (r: TrainingFinished) => void): Promise<UnlistenFn> =>
    listen<TrainingFinished>('training_finished', (e) => handler(e.payload)),

  onError: (handler: (err: TrainingError) => void): Promise<UnlistenFn> =>
    listen<TrainingError>('training_error', (e) => handler(e.payload)),
}
