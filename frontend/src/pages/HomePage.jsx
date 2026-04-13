import CaptureScreen from '../components/screens/CaptureScreen'
import ProcessingScreen from '../components/screens/ProcessingScreen'
import ResultsScreen from '../components/screens/ResultsScreen'
import { useSSEStream } from '../hooks/useSSEStream'
import { useAuth } from '../context/AuthContext'
import { useEffect } from 'react'

export default function HomePage() {
  const { uploading, processing, stages, results, error, done, process, reset, activeJobId } = useSSEStream()
  const { user } = useAuth()

  // Guest cleanup hook
  useEffect(() => {
    // We only clean up if NOT logged in, and we actually have a job ID.
    if (user || !activeJobId) return;

    const cleanupTarget = activeJobId

    // 1. Hook for when closing the tab/browser
    const handleBeforeUnload = () => {
      navigator.sendBeacon(`http://localhost:8000/api/cleanup/${cleanupTarget}`)
    }
    
    window.addEventListener('beforeunload', handleBeforeUnload)

    // 2. Hook for when React component unmounts normally
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
      fetch(`http://localhost:8000/api/cleanup/${cleanupTarget}`, { method: 'DELETE' }).catch(() => {})
    }
  }, [user, activeJobId])

  // Determine which screen is currently active
  const isProcessing = uploading || processing
  const isDone = done && results && results.length > 0

  // The active stage and active image number for the processing screen
  // stages is a record of { [idx]: 'ocr' | 'layout' | 'vlm' | 'export' | 'done' }
  const stageKeys = Object.keys(stages)
  const activeImageNumber = stageKeys.length > 0 ? stageKeys[stageKeys.length - 1] : 1
  const currentStage = stages[activeImageNumber] || 'uploading'

  return (
    <main className="min-h-[80vh] flex flex-col items-center justify-center p-4">
      {/* Screen 1: Initial Capture */}
      {!isProcessing && !isDone && (
        <CaptureScreen onSubmit={process} disabled={isProcessing} />
      )}

      {/* Screen 2: Processing Pipeline */}
      {isProcessing && !isDone && (
        <ProcessingScreen 
          stage={currentStage} 
          error={error} 
          activeImageNumber={activeImageNumber}
          maxTotal={Math.max(1, stageKeys.length)}
        />
      )}

      {/* Screen 3: Results */}
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
