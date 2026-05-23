import { useCallback, useRef, useState } from 'react'
import { getStreamUrl, uploadFiles } from '../services/api'

export function useSSEStream() {
  const [uploading, setUploading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [stages, setStages] = useState({})
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)
  const [activeJobId, setActiveJobId] = useState(null)
  const esRef = useRef(null)
  const isDoneRef = useRef(false)

  const reset = useCallback(() => {
    if (esRef.current) { esRef.current.close(); esRef.current = null }
    isDoneRef.current = false
    setUploading(false); setProcessing(false); setActiveJobId(null)
    setStages({}); setResults([]); setError(null); setDone(false)
  }, [])

  const process = useCallback(async (files) => {
    reset()
    isDoneRef.current = false
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
            isDoneRef.current = true
            setDone(true)
            setProcessing(false)
            es.close()
          }
        } catch (_) {}
      }

      es.onerror = () => {
        if (isDoneRef.current) return
        if (es.readyState === EventSource.CONNECTING) {
          console.warn("EventSource disconnected. Attempting to reconnect...")
          return
        }
        setError(`Connection to server lost (status: ${es.readyState}). Please try again.`)
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
