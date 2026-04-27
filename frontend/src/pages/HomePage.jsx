import CaptureScreen from '../components/screens/CaptureScreen'
import ProcessingScreen from '../components/screens/ProcessingScreen'
import ResultsScreen from '../components/screens/ResultsScreen'
import { useSSEStream } from '../hooks/useSSEStream'
import { useAuth } from '../context/AuthContext'
import { useEffect } from 'react'

export default function HomePage() {
  const { uploading, processing, stages, results, error, done, process, reset, activeJobId } = useSSEStream()
  const { user } = useAuth()

  useEffect(() => {
    if (user || !activeJobId) return;

    const cleanupTarget = activeJobId

    const handleBeforeUnload = () => {
      navigator.sendBeacon(`http://localhost:8000/api/cleanup/${cleanupTarget}`)
    }

    window.addEventListener('beforeunload', handleBeforeUnload)

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
      fetch(`http://localhost:8000/api/cleanup/${cleanupTarget}`, { method: 'DELETE' }).catch(() => {})
    }
  }, [user, activeJobId])

  const isProcessing = uploading || processing
  const isDone = done && results && results.length > 0

  const stageKeys = Object.keys(stages)
  const activeImageNumber = stageKeys.length > 0 ? stageKeys[stageKeys.length - 1] : 1
  const currentStage = stages[activeImageNumber] || 'uploading'

  return (
    <main className="w-full max-w-5xl mx-auto p-4 md:p-6">
      {!isProcessing && !isDone && (
        <CaptureScreen onSubmit={process} disabled={isProcessing} error={error} />
      )}

      {isProcessing && !isDone && (
        <ProcessingScreen
          stage={currentStage}
          error={error}
          activeImageNumber={activeImageNumber}
          maxTotal={Math.max(1, stageKeys.length)}
        />
      )}

      {isDone && (
        <ResultsScreen
          results={results}
          error={error}
          onReset={reset}
        />
      )}
    </main>
  )
}
