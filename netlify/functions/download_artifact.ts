import { getGithubEnv, gh, ghBuffer } from './_github'
import AdmZip from 'adm-zip'

type ArtifactsList = {
  artifacts: Array<{
    id: number
    name: string
    archive_download_url: string
    expired: boolean
  }>
}

type WorkflowRuns = {
  workflow_runs: Array<{
    id: number
    name: string | null
    display_title?: string | null
    status: 'queued' | 'in_progress' | 'completed'
    conclusion: 'success' | 'failure' | 'cancelled' | null
  }>
}

export async function handler(event: any) {
  if (event.httpMethod !== 'GET') {
    return { statusCode: 405, body: 'Method Not Allowed' }
  }

  const requestId = (event.queryStringParameters?.request_id || '').toString()
  if (!requestId) {
    return { statusCode: 400, body: 'Missing request_id' }
  }

  try {
    const { owner, repo, workflow, branch } = getGithubEnv()

    // Find the workflow run by run-name
    const runs = await gh<WorkflowRuns>(`/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(workflow)}/runs?event=repository_dispatch&branch=${encodeURIComponent(branch)}&per_page=50`)
    const name = `deck-${requestId}`
    const run = runs.workflow_runs.find(r => (r as any).display_title === name)
    if (!run) {
      return { statusCode: 404, body: 'Workflow run not found' }
    }

    // List artifacts on the run
    const arts = await gh<ArtifactsList>(`/repos/${owner}/${repo}/actions/runs/${run.id}/artifacts?per_page=100`)
    const art = arts.artifacts.find(a => a.name === `deck-${requestId}`)
    if (!art) {
      return { statusCode: 404, body: 'Artifact not found' }
    }

    // Download the artifact ZIP as a Buffer
    const zip = await ghBuffer(`/repos/${owner}/${repo}/actions/artifacts/${art.id}/zip`)
    const adm = new AdmZip(zip)
    const entries = adm.getEntries()
    const pptxEntry = entries.find(e => e.entryName.toLowerCase().endsWith('.pptx'))
    if (!pptxEntry) {
      return { statusCode: 404, body: 'PPTX not found in artifact' }
    }

    const pptx = pptxEntry.getData()
    const headers = {
      'Content-Type': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      'Content-Disposition': `attachment; filename="deck-${requestId}.pptx"`
    }
    return {
      statusCode: 200,
      headers,
      body: pptx.toString('base64'),
      isBase64Encoded: true
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return { statusCode: 500, body: message }
  }
}
