import { useState } from 'react'
import { deleteHistoryItem, fetchHistory } from '../services/api'

const SOURCE_META = {
  kaggle_remote_vlm: { label: 'Kaggle Qwen2.5-VL-32B (Remote Tunnel)', tier: '30GB VRAM' },
  llava_local_with_ocr_hint: { label: 'Local CUDA Qwen-VL (with OCR Hint)', tier: 'GPU' },
  llava_local_only: { label: 'Local CUDA Qwen-VL (Visual Only)', tier: 'GPU' },
  image_only_fallback: { label: 'Local Fallback (Visual Only)', tier: 'GPU' },
  llava: { label: 'LLaVA 1.6 Mistral-7B', tier: 'Tier 1' },
  blip2: { label: 'BLIP-2 Flan-T5-XL', tier: 'Tier 2' },
  local: { label: 'BLIP-base (local)', tier: 'Tier 3' },
  unavailable: { label: 'Model Unavailable', tier: '' },
}

function handleDownloadTxt(result) {
  const rawFields = result.vlm_fields || {}
  
  let text = `DOCUMENT EXTRACTION REPORT\n`
  text += `==========================\n\n`
  text += `File Name      : ${result.image_name || 'report'}\n`
  text += `Classification : ${result.metadata?.classification || 'Document'}\n`
  if (rawFields.metadata?.detected_language) {
    text += `Language       : ${rawFields.metadata.detected_language}\n`
  }
  text += `\n`

  if (rawFields.full_extraction) {
    text += `=========================================\n`
    text += `NATIVE SCRIPT OUTPUT\n`
    text += `=========================================\n`
    text += rawFields.full_extraction + `\n\n`
  }

  if (rawFields.english_extraction) {
    text += `=========================================\n`
    text += `ENGLISH TRANSLATION OUTPUT\n`
    text += `=========================================\n`
    text += rawFields.english_extraction + `\n\n`
  }

  if (result.text_report_preview) {
    text += `=========================================\n`
    text += `STRUCTURED REPORT PREVIEW\n`
    text += `=========================================\n`
    text += result.text_report_preview + `\n`
  }

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${result.image_name ? result.image_name.replace(/\.[^/.]+$/, "") : 'report'}_analysis_report.txt`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function ResultCard({ result, defaultOpen }) {
  const [expanded, setExpanded] = useState(defaultOpen)
  const success = result.status === 'success'

  const rawFields = result.vlm_fields || {}
  const source = result.vlm_source || 'unavailable'
  const meta = SOURCE_META[source] || SOURCE_META.unavailable

  return (
    <div className="border border-gray-200 rounded-md overflow-hidden bg-white shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 p-4 text-left hover:bg-gray-50 transition-colors"
      >
        <span
          className={`mt-1 h-2.5 w-2.5 rounded-full shrink-0 ${success ? 'bg-green-500' : 'bg-red-500'}`}
        ></span>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-gray-900 truncate">{result.image_name}</div>
          {result.error && <div className="text-xs text-red-600 mt-0.5">{result.error}</div>}
          {success && (
            <div className="flex flex-wrap gap-1.5 mt-1">
              <span className="tag capitalize">{result.metadata?.classification || 'Document'}</span>
              {result.metadata?.detected_language && (
                <span className="tag">{result.metadata.detected_language}</span>
              )}
              {result.vlm_source && result.vlm_source !== 'unavailable' && (
                <span className="tag">Kaggle Node</span>
              )}
            </div>
          )}
        </div>
        <span className="text-gray-500 text-xs shrink-0">{expanded ? 'Hide' : 'Show'}</span>
      </button>

      {expanded && success && (
        <div className="border-t border-gray-200 p-4 space-y-6">
          {/* VLM Inference Engine Info */}
          <div className="glass-light p-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">VLM Inference Engine</div>
              <p className="text-sm text-gray-900 font-semibold mt-0.5">{meta.label}</p>
            </div>
            <div className="flex gap-2">
              {rawFields.metadata?.detected_language && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
                  Lang: {rawFields.metadata.detected_language}
                </span>
              )}
              {meta.tier && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-50 text-indigo-800 border border-indigo-200">
                  {meta.tier}
                </span>
              )}
            </div>
          </div>

          {/* Download TXT Report Section */}
          {result.text_report_preview && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 bg-blue-50/50 border border-blue-100 rounded-lg">
              <div>
                <div className="font-semibold text-blue-900 text-sm">Download Report</div>
                <div className="text-xs text-blue-700 mt-0.5">Save the structured text report of the document.</div>
              </div>
              <button
                onClick={() => handleDownloadTxt(result)}
                className="w-full sm:w-auto flex items-center justify-center gap-2 bg-gray-900 text-white font-medium px-4 py-2 rounded-md hover:bg-gray-800 transition-colors shadow-sm text-sm"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download TXT
              </button>
            </div>
          )}

          {/* Report Preview */}
          {result.text_report_preview && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="h-4 w-1 bg-gray-500 rounded-full"></div>
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider">Report Preview</h3>
              </div>
              <pre className="bg-gray-50 p-4 rounded-lg border border-gray-200 text-[11px] text-gray-700 font-mono whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto shadow-inner">
                {result.text_report_preview}
              </pre>
            </div>
          )}

          {/* Spatial OCR Reconstructions */}
          {rawFields && (rawFields.full_extraction || rawFields.english_extraction) && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="h-4 w-1 bg-gray-900 rounded-full"></div>
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider">Spatial OCR Reconstructions</h3>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {rawFields.full_extraction && (
                  <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <h4 className="text-xs font-bold text-gray-500 mb-2 uppercase tracking-wide">Native Script Output</h4>
                    <pre className="text-[11px] text-gray-800 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto max-h-60 overflow-y-auto">
                      {rawFields.full_extraction}
                    </pre>
                  </div>
                )}
                {rawFields.english_extraction && (
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <h4 className="text-xs font-bold text-blue-600 mb-2 uppercase tracking-wide">English Translation Output</h4>
                    <pre className="text-[11px] text-blue-900 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto max-h-60 overflow-y-auto">
                      {rawFields.english_extraction}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function HistoryRow({ item, onDelete }) {
  return (
    <div className="glass-light px-3 py-2.5 flex items-center gap-3 border border-gray-200">
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-900 truncate">{item.file_name}</div>
        <div className="text-[11px] text-gray-500 mt-0.5">
          {item.created_at ? new Date(item.created_at).toLocaleString() : ''}
        </div>
      </div>
      <span className="tag capitalize shrink-0">{item.document_type || '—'}</span>
      <span className={`tag shrink-0 ${item.status === 'success' ? 'text-green-700 bg-green-50 border-green-200' : 'text-red-700 bg-red-50 border-red-200'}`}>
        {item.status}
      </span>
      <button onClick={() => onDelete(item._id)} className="btn-ghost text-xs py-1 px-2 border border-gray-300 rounded hover:bg-gray-100 transition-colors">
        Delete
      </button>
    </div>
  )
}

export default function ResultTabs({ results }) {
  const [history, setHistory] = useState(null)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)

  const loadHistory = async () => {
    setLoadingHistory(true)
    try {
      const data = await fetchHistory(20)
      setHistory(data.results || [])
    } catch (_) {
      setHistory([])
    } finally {
      setLoadingHistory(false)
    }
  }

  const handleDelete = async (id) => {
    try {
      await deleteHistoryItem(id)
      setHistory(prev => prev ? prev.filter(h => h._id !== id) : null)
    } catch (err) {
      console.error("Failed to delete history item:", err)
    }
  }

  if (!results || results.length === 0) return null

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">Results ({results.length})</h2>
        {results.map((r, i) => (
          <ResultCard key={i} result={r} defaultOpen={i === 0} />
        ))}
      </div>

      {/* Global Past Extractions Section */}
      <div className="border-t border-gray-200 pt-6">
        <button
          onClick={() => {
            const nextOpen = !historyOpen
            setHistoryOpen(nextOpen)
            if (nextOpen && !history) {
              loadHistory()
            }
          }}
          className="flex items-center justify-between w-full p-3 bg-gray-50 border border-gray-200 rounded-md hover:bg-gray-100 transition-colors"
        >
          <span className="text-sm font-semibold text-gray-700">Past Extractions History</span>
          <span className="text-xs text-gray-500">{historyOpen ? 'Hide' : 'Show'}</span>
        </button>

        {historyOpen && (
          <div className="mt-3 p-4 bg-white border border-gray-200 rounded-md space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Showing last 20 extractions</span>
              <button
                onClick={loadHistory}
                className="btn-ghost text-xs py-1 px-3 border border-gray-300 rounded hover:bg-gray-100 transition-colors"
                disabled={loadingHistory}
              >
                {loadingHistory ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {loadingHistory && !history ? (
              <div className="text-center py-4 text-sm text-gray-500">Loading history...</div>
            ) : history ? (
              history.length === 0 ? (
                <p className="text-sm text-gray-500 italic text-center py-4">No history found.</p>
              ) : (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {history.map(h => (
                    <HistoryRow key={h._id} item={h} onDelete={handleDelete} />
                  ))}
                </div>
              )
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}
