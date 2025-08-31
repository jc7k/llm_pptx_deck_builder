import { getGithubEnv, gh } from './_github'

type CreateJobBody = {
  request_id?: string
  topic?: string
  slide_count?: number
  openai_api_key?: string
  brave_api_key?: string
}

export async function handler(event: any) {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' }
  }

  try {
    const body = JSON.parse(event.body || '{}') as CreateJobBody
    const requestId = (body.request_id || '').toString()
    const topic = (body.topic || '').toString().trim()
    const slideCount = body.slide_count
    const openai = (body.openai_api_key || '').toString().trim()
    const brave = (body.brave_api_key || '').toString().trim()
    if (!requestId || !topic || !openai || !brave) {
      return { statusCode: 400, body: 'Missing required fields' }
    }

    const { owner, repo } = getGithubEnv()
    await gh<void>(`/repos/${owner}/${repo}/dispatches`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_type: 'deck-request',
        client_payload: {
          request_id: requestId,
          topic,
          slide_count: slideCount != null ? Number(slideCount) : 12,
          openai_api_key: openai,
          brave_api_key: brave
        }
      })
    })

    return {
      statusCode: 202,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: requestId })
    }
  } catch (err) {
    return { statusCode: 500, body: 'Failed to create job' }
  }
}
