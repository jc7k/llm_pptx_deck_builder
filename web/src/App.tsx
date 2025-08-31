import { useEffect, useMemo, useRef, useState } from 'react'
import { createJob, downloadArtifact, getJobStatus } from './api'
import type { JobStatus } from './types'

type Phase = 'idle' | 'submitting' | 'polling' | 'ready' | 'error'

export function App() {
  const [topic, setTopic] = useState('')
  const [slideCount, setSlideCount] = useState<number | ''>('')
  const [openaiKey, setOpenaiKey] = useState('')
  const [braveKey, setBraveKey] = useState('')
  const [phase, setPhase] = useState<Phase>('idle')
  const [requestId, setRequestId] = useState<string | null>(null)
  const [status, setStatus] = useState<JobStatus>({ state: 'not_found' })
  const [error, setError] = useState<string | null>(null)

  const canSubmit = useMemo(
    () =>
      topic.trim().length > 0 &&
      openaiKey.trim().length > 0 &&
      braveKey.trim().length > 0 &&
      phase !== 'submitting' &&
      phase !== 'polling',
    [topic, openaiKey, braveKey, phase]
  )
  const pollTimer = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (pollTimer.current) window.clearTimeout(pollTimer.current)
    }
  }, [])

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setPhase('submitting')

    try {
      const rid = crypto.randomUUID()
      setRequestId(rid)

      await createJob({
        request_id: rid,
        topic: topic.trim(),
        slide_count: slideCount === '' ? undefined : Number(slideCount),
        openai_api_key: openaiKey.trim(),
        brave_api_key: braveKey.trim(),
      })

      // Clear sensitive keys from memory after dispatch
      setOpenaiKey('')
      setBraveKey('')

      setPhase('polling')
      void pollStatus(rid, 0)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setPhase('error')
    }
  }

  async function pollStatus(rid: string, attempt: number) {
    try {
      const s = await getJobStatus(rid)
      setStatus(s)
      if (s.state === 'queued' || s.state === 'in_progress' || s.state === 'not_found') {
        const delay = Math.min(6000, 2000 + attempt * 500)
        pollTimer.current = window.setTimeout(() => void pollStatus(rid, attempt + 1), delay)
      } else if (s.state === 'completed' && s.conclusion === 'success') {
        setPhase('ready')
      } else if (s.state === 'completed') {
        setPhase('error')
        setError('The build completed without success.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setPhase('error')
    }
  }

  async function onDownload() {
    if (!requestId) return
    try {
      const blob = await downloadArtifact(requestId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `deck-${requestId}.pptx`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: '2rem auto', padding: '0 1rem', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Deck Builder</h1>
      <form onSubmit={onSubmit} style={{ display: 'grid', gap: '0.75rem' }}>
        <fieldset style={{ border: '1px solid #ddd', padding: '0.75rem' }}>
          <legend>Bring your own API keys</legend>
          <p style={{ marginTop: 0, fontSize: '0.9em', color: '#555' }}>
            Keys are used only for this session in-memory, never stored or logged. Do not share production keys.
          </p>
          <label>
            <div>OpenAI API Key</div>
            <input
              type="password"
              inputMode="text"
              autoComplete="off"
              spellCheck={false}
              value={openaiKey}
              onChange={(e) => setOpenaiKey(e.target.value)}
              style={{ width: '100%', padding: '0.5rem' }}
            />
          </label>
          <label>
            <div>Brave API Key</div>
            <input
              type="password"
              inputMode="text"
              autoComplete="off"
              spellCheck={false}
              value={braveKey}
              onChange={(e) => setBraveKey(e.target.value)}
              style={{ width: '100%', padding: '0.5rem' }}
            />
          </label>
        </fieldset>
        <label>
          <div>Topic</div>
          <input
            type="text"
            value={topic}
            placeholder="e.g., AI in 2025"
            onChange={(e) => setTopic(e.target.value)}
            style={{ width: '100%', padding: '0.5rem' }}
          />
        </label>
        <label>
          <div>Slide count (optional)</div>
          <input
            type="number"
            min={5}
            max={40}
            step={1}
            value={slideCount}
            onChange={(e) => setSlideCount(e.target.value === '' ? '' : Number(e.target.value))}
            style={{ width: '100%', padding: '0.5rem' }}
          />
        </label>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button type="submit" disabled={!canSubmit}>
            {phase === 'submitting' ? 'Submitting…' : phase === 'polling' ? 'Queued/Running…' : 'Generate Deck'}
          </button>
          {phase === 'ready' && (
            <button type="button" onClick={onDownload}>
              Download deck
            </button>
          )}
        </div>
      </form>

      <div style={{ marginTop: '1rem' }}>
        <strong>Status:</strong>{' '}
        {(() => {
          switch (status.state) {
            case 'queued':
              return 'Queued'
            case 'in_progress':
              return 'In progress'
            case 'completed':
              return `Completed (${status.conclusion})`
            case 'not_found':
              return phase === 'idle' ? 'Idle' : 'Waiting for run…'
          }
        })()}
      </div>

      {requestId && (
        <div style={{ marginTop: '0.5rem', fontSize: '0.9em', color: '#555' }}>
          <code>request_id: {requestId}</code>
        </div>
      )}

      {error && (
        <div style={{ marginTop: '1rem', color: '#b00020' }}>
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  )
}
