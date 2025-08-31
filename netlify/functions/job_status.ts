import { getGithubEnv, gh } from './_github'

type WorkflowRuns = {
  workflow_runs: Array<{
    id: number
    name: string | null
    display_title?: string | null
    run_number: number
    run_attempt: number
    status: 'queued' | 'in_progress' | 'completed'
    conclusion: 'success' | 'failure' | 'cancelled' | null
    run_started_at: string
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
    const runs = await gh<WorkflowRuns>(`/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(workflow)}/runs?event=repository_dispatch&branch=${encodeURIComponent(branch)}&per_page=50`, {
      method: 'GET'
    })

    const name = `deck-${requestId}`
    const run = runs.workflow_runs.find(r => (r as any).display_title === name)

    if (!run) {
      return json({ state: 'not_found' })
    }

    if (run.status === 'completed') {
      return json({ state: 'completed', conclusion: run.conclusion || 'failure', run_id: run.id })
    }

    return json({ state: run.status })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return { statusCode: 500, body: message }
  }
}

function json(body: any) {
  return { statusCode: 200, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}
