import { useEffect, useState, useCallback } from 'react'
import CaptureScreen from '../components/screens/CaptureScreen'
import ProcessingScreen from '../components/screens/ProcessingScreen'
import ResultsScreen from '../components/screens/ResultsScreen'
import { useSSEStream } from '../hooks/useSSEStream'
import { useAuth } from '../context/AuthContext'
import { getFileUrl, fetchHistory } from '../services/api'

export default function HomePage() {
  const { uploading, processing, stages, results, error, done, process, reset, activeJobId } = useSSEStream()
  const { user, token } = useAuth()
  
  const [history, setHistory] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [selectedHistoryResult, setSelectedHistoryResult] = useState(null)

  const loadHistory = useCallback(() => {
    if (!token) return
    setLoadingHistory(true)
    fetchHistory(20, token)
      .then(data => {
        setHistory(data.results || [])
      })
      .catch(err => console.error("Failed to load history:", err))
      .finally(() => setLoadingHistory(false))
  }, [token])

  useEffect(() => {
    if (user && token) {
      loadHistory()
    } else {
      setHistory([])
      setSelectedHistoryResult(null)
    }
  }, [user, token, loadHistory])

  const isProcessing = uploading || processing
  const isDone = (done && results && results.length > 0) || !!selectedHistoryResult
  const displayResults = selectedHistoryResult ? [selectedHistoryResult] : results

  // Refresh history list when a new upload successfully completes
  useEffect(() => {
    const freshDone = done && results && results.length > 0
    if (freshDone) {
      loadHistory()
    }
  }, [done, results, loadHistory])

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

  const handleReset = () => {
    setSelectedHistoryResult(null)
    reset()
  }

  const handleViewHistoryItem = (item) => {
    const mapped = {
      ...item,
      image_name: item.file_name || item.image_name,
      vlm_fields: item.vlm?.fields || item.vlm_fields || {},
      vlm_source: item.vlm?.source || item.vlm_source || 'unavailable',
      metadata: {
        classification: item.document_type || item.metadata?.classification || 'Document',
        detected_language: item.ocr?.language_summary?.detected_language || item.metadata?.detected_language
      }
    }
    setSelectedHistoryResult(mapped)
  }

  const stageKeys = Object.keys(stages)
  const activeImageNumber = stageKeys.length > 0 ? stageKeys[stageKeys.length - 1] : 1
  const currentStage = stages[activeImageNumber] || 'uploading'

  return (
    <main className="w-full max-w-5xl mx-auto p-4 md:p-6 space-y-6">
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
          results={displayResults}
          error={error}
          onReset={handleReset}
        />
      )}

      {/* History panel visible below upload scanner for logged-in users */}
      {!isProcessing && !isDone && user && (
        <section className="glass p-6 space-y-4">
          <div className="flex items-center gap-2">
            <div className="h-5 w-1 bg-gray-900 rounded-full"></div>
            <h2 className="text-lg font-bold text-gray-900">Your Extraction History</h2>
          </div>
          {loadingHistory ? (
            <p className="text-sm text-gray-500 animate-pulse">Loading history...</p>
          ) : history.length === 0 ? (
            <p className="text-sm text-gray-500">No past extractions found. Upload an invoice to get started!</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-gray-600">
                <thead>
                  <tr className="border-b border-gray-200 text-xs font-bold uppercase text-gray-500">
                    <th className="py-2.5">File Name</th>
                    <th className="py-2.5">Classification</th>
                    <th className="py-2.5">Date</th>
                    <th className="py-2.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {history.map((item, idx) => {
                    const name = item.file_name || item.image_name || 'unknown'
                    const docType = item.document_type || item.metadata?.classification || 'Document'
                    const date = item.created_at || item.timestamp
                    
                    return (
                      <tr key={idx} className="hover:bg-gray-50 transition-colors">
                        <td className="py-3 font-medium text-gray-900 max-w-[200px] truncate" title={name}>
                          {name}
                        </td>
                        <td className="py-3 capitalize">
                          {docType}
                        </td>
                        <td className="py-3 text-xs text-gray-500">
                          {date ? new Date(date).toLocaleString() : 'N/A'}
                        </td>
                        <td className="py-3 text-right">
                          <button
                            onClick={() => handleViewHistoryItem(item)}
                            className="text-xs text-indigo-600 hover:text-indigo-800 font-bold underline"
                          >
                            View Results
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </main>
  )
}

