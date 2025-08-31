export type CreateJobRequest = {
  request_id: string
  topic: string
  slide_count?: number
  openai_api_key: string
  brave_api_key: string
}

export type CreateJobResponse = {
  request_id: string
}

export type JobStatus =
  | { state: 'not_found' }
  | { state: 'queued' }
  | { state: 'in_progress' }
  | { state: 'completed'; conclusion: 'success' | 'failure' | 'cancelled'; run_id: number }
