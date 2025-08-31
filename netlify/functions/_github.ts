const BASE = 'https://api.github.com'

export type GithubEnv = {
  owner: string
  repo: string
  workflow: string
  branch: string
  token: string
}

export function getGithubEnv(): GithubEnv {
  const owner = process.env.GITHUB_OWNER
  const repo = process.env.GITHUB_REPO
  const workflow = process.env.GITHUB_WORKFLOW || 'build_deck.yml'
  const branch = process.env.GITHUB_BRANCH || 'main'
  const token = process.env.GITHUB_TOKEN
  if (!owner || !repo || !token) {
    throw new Error('Missing required env: GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN')
  }
  return { owner, repo, workflow, branch, token }
}

export async function gh<T>(path: string, init?: RequestInit): Promise<T> {
  const { token } = getGithubEnv()
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${token}`,
    'Accept': 'application/vnd.github+json',
    'User-Agent': 'netlify-functions-deck-builder'
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers: { ...headers, ...(init?.headers as any) } })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`GitHub API ${res.status} ${res.statusText}: ${text}`)
  }
  // Some endpoints return 204 No Content
  const ct = res.headers.get('content-type') || ''
  if (res.status === 204 || !ct.includes('application/json')) {
    return undefined as unknown as T
  }
  return res.json() as Promise<T>
}

export async function ghBuffer(path: string, init?: RequestInit): Promise<Buffer> {
  const { token } = getGithubEnv()
  const headers: Record<string, string> = {
    'Authorization': `Bearer ${token}`,
    'Accept': 'application/vnd.github+json',
    'User-Agent': 'netlify-functions-deck-builder'
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers: { ...headers, ...(init?.headers as any) } })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`GitHub API ${res.status} ${res.statusText}: ${text}`)
  }
  const ab = await res.arrayBuffer()
  return Buffer.from(ab)
}

