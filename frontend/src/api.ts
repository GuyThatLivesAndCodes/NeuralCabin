import axios from 'axios'

const API_BASE = 'http://localhost:3001'

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
  last_val_loss?: number
  last_accuracy?: number
  loss_history: number[]
  val_loss_history: number[]
  accuracy_history: number[]
  elapsed_secs: number
}

export interface WsMessage {
  type: 'epoch_update' | 'training_finished' | 'error'
  epoch?: number
  total_epochs?: number
  last_loss?: number
  last_val_loss?: number
  last_accuracy?: number
  loss_history?: number[]
  val_loss_history?: number[]
  accuracy_history?: number[]
  elapsed_secs?: number
  status?: string
  final_loss?: number
  message?: string
}

const api = axios.create({ baseURL: API_BASE })

// Networks
export const networks = {
  create: (data: {
    name: string
    kind: string
    seed: number
    layers: Layer[]
  }) => api.post<Network>('/api/networks', data),

  list: () => api.get<{ networks: Network[] }>('/api/networks'),

  get: (id: string) => api.get<Network>(`/api/networks/${id}`),

  delete: (id: string) => api.delete(`/api/networks/${id}`),
}

// Datasets
export const datasets = {
  create: (data: { name: string; kind: string; seed: number }) =>
    api.post<Dataset>('/api/datasets', data),

  list: () => api.get<{ datasets: Dataset[] }>('/api/datasets'),
}

// Training
export const training = {
  start: (data: TrainingRequest) =>
    api.post<{ training_id: string; status: string }>('/api/train', data),

  status: (trainingId: string) =>
    api.get<TrainingStatus>(`/api/train/${trainingId}`),

  connect: (trainingId: string): WebSocket => {
    return new WebSocket(`ws://localhost:3001/ws/train/${trainingId}`)
  },
}
