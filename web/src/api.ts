import type { CreateJobRequest, CreateJobResponse, JobStatus } from './types'

const base = '' // relative to current origin

export async function createJob(input: CreateJobRequest): Promise<CreateJobResponse> {
  const res = await fetch(`${base}/.netlify/functions/create_job`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!res.ok) {
    throw new Error(`create_job failed: ${res.status}`)
  }
  return res.json()
}

export async function getJobStatus(requestId: string): Promise<JobStatus> {
  const res = await fetch(`${base}/.netlify/functions/job_status?request_id=${encodeURIComponent(requestId)}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`job_status failed: ${res.status} ${text}`)
  }
  return res.json()
}

export async function downloadArtifact(requestId: string): Promise<Blob> {
  const res = await fetch(`${base}/.netlify/functions/download_artifact?request_id=${encodeURIComponent(requestId)}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`download_artifact failed: ${res.status} ${text}`)
  }
  return res.blob()
}
