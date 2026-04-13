// frontend/src/hooks/useSSEStream.js
import { useCallback, useRef, useState } from 'react'
import { getStreamUrl, uploadFiles } from '../services/api'

export function useSSEStream() {
  const [uploading, setUploading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [stages, setStages] = useState({})       // { imageIdx: currentStage }
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)
  const [activeJobId, setActiveJobId] = useState(null)
  const esRef = useRef(null)

  const reset = useCallback(() => {
    if (esRef.current) { esRef.current.close(); esRef.current = null }
    setUploading(false); setProcessing(false); setActiveJobId(null)
    setStages({}); setResults([]); setError(null); setDone(false)
  }, [])

  const process = useCallback(async (files) => {
    reset()
    setUploading(true)
    setError(null)

    try {
      const token = localStorage.getItem('token')
      const { job_id } = await uploadFiles(files, token)
      setActiveJobId(job_id)
      setUploading(false)
      setProcessing(true)

      const es = new EventSource(getStreamUrl(job_id))
      esRef.current = es

      es.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.event === 'stage') {
            setStages(prev => ({ ...prev, [msg.image]: msg.stage }))
          } else if (msg.event === 'result') {
            setResults(prev => [...prev, { ...msg.data, _imageIdx: msg.image }])
          } else if (msg.event === 'done') {
            setDone(true)
            setProcessing(false)
            es.close()
          }
        } catch (_) {}
      }

      es.onerror = () => {
        setError('Connection to server lost. Please try again.')
        setProcessing(false)
        es.close()
      }
    } catch (err) {
      setUploading(false)
      setProcessing(false)
      setError(err?.response?.data?.detail || err.message || 'Upload failed')
    }
  }, [reset])

  return { uploading, processing, stages, results, error, done, process, reset, activeJobId }
}
