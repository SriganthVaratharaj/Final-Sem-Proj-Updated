import { useState } from 'react'
import { deleteHistoryItem, fetchHistory } from '../../services/api'

function DownloadBtn({ href, label }) {
  const hasLink = href && href !== ''

  return (
    <a
      href={hasLink ? href : undefined}
      download
      target="_blank"
      rel="noopener noreferrer"
      className={`block p-3 rounded-md border text-sm transition-colors ${
        hasLink
          ? 'border-gray-300 text-gray-900 hover:bg-gray-50'
          : 'border-gray-200 text-gray-400 cursor-not-allowed'
      }`}
    >
      <div className="font-medium">{label}</div>
      <div className="text-xs mt-0.5">{hasLink ? 'Click to download' : 'Not available'}</div>
    </a>
  )
}

function HistoryRow({ item, onDelete }) {
  return (
    <div className="glass-light px-3 py-2 flex items-center gap-3">
      <div className="flex-1 min-w-0">
        <div className="text-sm text-gray-900 truncate">{item.file_name}</div>
        <div className="text-xs text-gray-500">{item.created_at ? new Date(item.created_at).toLocaleString() : ''}</div>
      </div>
      <span className="tag capitalize shrink-0">{item.document_type || '—'}</span>
      <span className={`tag shrink-0 ${item.status === 'success' ? 'text-green-700 bg-green-50 border-green-200' : 'text-red-700 bg-red-50 border-red-200'}`}>
        {item.status}
      </span>
      <button onClick={() => onDelete(item._id)} className="btn-ghost text-xs py-1 px-2">
        Delete
      </button>
    </div>
  )
}

export default function ExportsTab({ result }) {
  const [history, setHistory] = useState(null)
  const [loadingHistory, setLoadingHistory] = useState(false)

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
    await deleteHistoryItem(id)
    setHistory(prev => prev.filter(h => h._id !== id))
  }

  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Download Exports</h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
          <DownloadBtn href={result.excel_file_url} label="Excel (.xlsx)" />
          <DownloadBtn href={result.json_output_url} label="JSON Data" />
          <DownloadBtn href={result.digital_twin_txt_url} label="Digital Twin (.txt)" />
          <DownloadBtn href={result.digital_twin_docx_url} label="Digital Twin (.docx)" />
        </div>
      </div>

      {result.text_report_preview && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Report Preview</h3>
          <pre className="glass-light p-3 text-xs text-gray-700 font-mono whitespace-pre-wrap leading-relaxed max-h-52 overflow-y-auto">
            {result.text_report_preview}
          </pre>
        </div>
      )}

      <div className="glass-light px-3 py-2 text-sm text-gray-700">
        Google Sheets: {result.gsheets_synced ? 'Synced' : 'Not synced'}
      </div>

      {result.db_id && (
        <div className="text-xs text-gray-600">
          <span className="field-label">MongoDB ID:</span> <span className="font-mono">{result.db_id}</span>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-700">Past Extractions</h3>
          <button onClick={loadHistory} className="btn-ghost text-xs py-1 px-3" disabled={loadingHistory}>
            {loadingHistory ? 'Loading...' : 'Load history'}
          </button>
        </div>
        {history && (
          history.length === 0
            ? <p className="text-sm text-gray-500 italic">No history found.</p>
            : <div className="space-y-1.5 max-h-64 overflow-y-auto">{history.map(h => <HistoryRow key={h._id} item={h} onDelete={handleDelete} />)}</div>
        )}
      </div>
    </div>
  )
}
