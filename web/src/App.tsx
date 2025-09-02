import { useEffect, useMemo, useRef, useState } from 'react'
import { createJob, downloadArtifact, getJobStatus } from './api'
import type { JobStatus } from './types'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Progress } from './components/ui/progress'
import { Download, Loader2, FileText, Sparkles, Shield, Key } from 'lucide-react'

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

  const getProgressValue = () => {
    if (phase === 'submitting') return 25
    if (phase === 'polling') {
      if (status.state === 'queued') return 50
      if (status.state === 'in_progress') return 75
    }
    if (phase === 'ready') return 100
    return 0
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="container max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-primary rounded-xl">
              <FileText className="w-8 h-8 text-primary-foreground" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 bg-clip-text text-transparent dark:from-slate-100 dark:to-slate-400">
              AI Deck Builder
            </h1>
            <Sparkles className="w-8 h-8 text-amber-500" />
          </div>
          <p className="text-lg text-muted-foreground">
            Generate professional PowerPoint presentations with AI-powered research and citations
          </p>
        </div>

        <form onSubmit={onSubmit} className="space-y-8">
          {/* API Keys Card */}
          <Card className="border-amber-200 bg-amber-50/50 dark:border-amber-900 dark:bg-amber-950/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-amber-600" />
                API Configuration
              </CardTitle>
              <CardDescription>
                Your API keys are used only for this session and never stored or logged. 
                Use development keys only - never share production credentials.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="openai-key" className="flex items-center gap-2">
                    <Key className="w-4 h-4" />
                    OpenAI API Key
                  </Label>
                  <Input
                    id="openai-key"
                    type="password"
                    inputMode="text"
                    autoComplete="off"
                    spellCheck={false}
                    value={openaiKey}
                    onChange={(e) => setOpenaiKey(e.target.value)}
                    placeholder="sk-..."
                    className="font-mono"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="brave-key" className="flex items-center gap-2">
                    <Key className="w-4 h-4" />
                    Brave Search API Key
                  </Label>
                  <Input
                    id="brave-key"
                    type="password"
                    inputMode="text"
                    autoComplete="off"
                    spellCheck={false}
                    value={braveKey}
                    onChange={(e) => setBraveKey(e.target.value)}
                    placeholder="brv-..."
                    className="font-mono"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Content Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Presentation Details</CardTitle>
              <CardDescription>
                Configure your AI-generated presentation topic and parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="topic">Presentation Topic</Label>
                <Input
                  id="topic"
                  type="text"
                  value={topic}
                  placeholder="e.g., Future of Artificial Intelligence in Healthcare"
                  onChange={(e) => setTopic(e.target.value)}
                  className="text-lg"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="slide-count">
                  Number of Slides <span className="text-muted-foreground">(optional, 5-40)</span>
                </Label>
                <Input
                  id="slide-count"
                  type="number"
                  min={5}
                  max={40}
                  step={1}
                  value={slideCount}
                  onChange={(e) => setSlideCount(e.target.value === '' ? '' : Number(e.target.value))}
                  placeholder="Leave empty for AI to decide"
                  className="w-full md:w-48"
                />
              </div>
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              type="submit" 
              disabled={!canSubmit}
              size="lg"
              className="text-lg px-8 py-6"
            >
              {phase === 'submitting' && <Loader2 className="w-5 h-5 mr-2 animate-spin" />}
              {phase === 'polling' && <Loader2 className="w-5 h-5 mr-2 animate-spin" />}
              {phase === 'submitting' ? 'Submitting...' : 
               phase === 'polling' ? 'Generating...' : 
               'Generate Presentation'}
            </Button>
            {phase === 'ready' && (
              <Button 
                type="button" 
                onClick={onDownload}
                variant="outline"
                size="lg"
                className="text-lg px-8 py-6"
              >
                <Download className="w-5 h-5 mr-2" />
                Download PowerPoint
              </Button>
            )}
          </div>
        </form>

        {/* Status Card */}
        {(phase !== 'idle' || error) && (
          <Card className="mt-8">
            <CardHeader>
              <CardTitle>Generation Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Progress Bar */}
              {(phase === 'submitting' || phase === 'polling') && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{getProgressValue()}%</span>
                  </div>
                  <Progress value={getProgressValue()} className="h-2" />
                </div>
              )}

              {/* Status Text */}
              <div className="flex items-center gap-2">
                <strong>Status:</strong>
                <span className={`px-2 py-1 rounded-md text-sm font-medium ${
                  status.state === 'completed' && status.conclusion === 'success' 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100'
                    : status.state === 'completed' && status.conclusion !== 'success'
                    ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100'
                    : status.state === 'in_progress'
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100'
                    : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100'
                }`}>
                  {(() => {
                    switch (status.state) {
                      case 'queued':
                        return 'Queued'
                      case 'in_progress':
                        return 'Generating presentation...'
                      case 'completed':
                        return status.conclusion === 'success' ? 'Ready for download!' : `Failed: ${status.conclusion}`
                      case 'not_found':
                        return phase === 'idle' ? 'Ready to generate' : 'Initializing...'
                    }
                  })()}
                </span>
              </div>

              {/* Request ID */}
              {requestId && (
                <div className="text-sm text-muted-foreground font-mono bg-muted p-2 rounded">
                  Request ID: {requestId}
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-center gap-2 text-destructive">
                    <strong>Error:</strong>
                    <span>{error}</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
