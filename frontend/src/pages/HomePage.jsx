import { useState, useEffect } from 'react'
import CaptureScreen from '../components/screens/CaptureScreen'
import ProcessingScreen from '../components/screens/ProcessingScreen'
import ResultsScreen from '../components/screens/ResultsScreen'
import Dashboard from '../components/Dashboard'
import { useSSEStream } from '../hooks/useSSEStream'
import { useAuth } from '../context/AuthContext'
import { getFileUrl } from '../services/api'

export default function HomePage() {
  const { uploading, processing, stages, results, error, done, process, reset, activeJobId } = useSSEStream()
  const { user } = useAuth()
  const [activeTab, setActiveTab] = useState('upload')

  useEffect(() => {
    if (user || !activeJobId) return;

    const cleanupTarget = activeJobId

    const handleBeforeUnload = () => {
      // sendBeacon only supports POST; use keepalive fetch for DELETE
      fetch(getFileUrl(`/api/cleanup/${cleanupTarget}`), {
        method: 'DELETE',
        keepalive: true,
      }).catch(() => {})
    }

    window.addEventListener('beforeunload', handleBeforeUnload)

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
      fetch(getFileUrl(`/api/cleanup/${cleanupTarget}`), { method: 'DELETE' }).catch(() => {})
    }
  }, [user, activeJobId])

  // Automatically switch back to upload view if user signs out
  useEffect(() => {
    if (!user) {
      setActiveTab('upload')
    }
  }, [user])

  const isProcessing = uploading || processing
  const isDone = done && results && results.length > 0

  const stageKeys = Object.keys(stages)
  const activeImageNumber = stageKeys.length > 0 ? stageKeys[stageKeys.length - 1] : 1
  const currentStage = stages[activeImageNumber] || 'uploading'

  return (
    <main className="w-full max-w-5xl mx-auto p-4 md:p-6 space-y-6">
      {/* Dynamic Tab Navigation for logged-in users */}
      {user && !isProcessing && !isDone && (
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-5 py-2.5 text-sm font-semibold border-b-2 transition-colors duration-150 ${
              activeTab === 'upload'
                ? 'border-gray-900 text-gray-900'
                : 'border-transparent text-gray-500 hover:text-gray-800'
            }`}
          >
            Invoice Extractor
          </button>
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-5 py-2.5 text-sm font-semibold border-b-2 transition-colors duration-150 ${
              activeTab === 'dashboard'
                ? 'border-gray-900 text-gray-900'
                : 'border-transparent text-gray-500 hover:text-gray-800'
            }`}
          >
            Analytics & Search Dashboard
          </button>
        </div>
      )}

      {/* Render selected screen based on active tab state */}
      {activeTab === 'dashboard' && user ? (
        <Dashboard />
      ) : (
        <>
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
        </>
      )}
    </main>
  )
}
