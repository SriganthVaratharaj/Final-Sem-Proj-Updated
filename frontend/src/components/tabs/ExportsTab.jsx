// frontend/src/components/tabs/ExportsTab.jsx
import { Download, FileSpreadsheet, FileText, FileJson, CheckCircle, XCircle, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { deleteHistoryItem, fetchHistory } from '../../services/api'

function DownloadBtn({ href, label, icon: Icon, color }) {
  const hasLink = href && href !== ''
  return (
    <a
      href={hasLink ? href : undefined}
      download
      target="_blank"
      rel="noopener noreferrer"
      className={`flex items-center gap-3 glass-light p-4 rounded-xl border transition-all duration-200
        ${hasLink
          ? `border-${color}-700/40 hover:border-${color}-500 hover:bg-${color}-900/20 cursor-pointer`
          : 'border-gray-800 opacity-40 cursor-not-allowed'
        }`}
    >
      <Icon size={20} className={hasLink ? `text-${color}-400` : 'text-gray-600'} />
      <div>
        <div className={`text-sm font-semibold ${hasLink ? 'text-white' : 'text-gray-600'}`}>{label}</div>
        <div className="text-xs text-gray-500">{hasLink ? 'Click to download' : 'Not available'}</div>
      </div>
      {hasLink && <Download size={14} className={`ml-auto text-${color}-400`} />}
    </a>
  )
}

function HistoryRow({ item, onDelete }) {
  return (
    <div className="glass-light px-4 py-3 flex items-center gap-3">
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-white truncate">{item.file_name}</div>
        <div className="text-xs text-gray-500">{item.created_at ? new Date(item.created_at).toLocaleString() : ''}</div>
      </div>
      <span className="tag bg-gray-800 text-gray-400 capitalize shrink-0">{item.document_type || '—'}</span>
      <span className={`tag shrink-0 ${item.status === 'success' ? 'bg-emerald-900/40 text-emerald-400' : 'bg-red-900/40 text-red-400'}`}>
        {item.status}
      </span>
      <button onClick={() => onDelete(item._id)} className="text-gray-700 hover:text-red-400 shrink-0 transition-colors">
        <Trash2 size={14} />
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
    } catch (_) { setHistory([]) }
    finally { setLoadingHistory(false) }
  }

  const handleDelete = async (id) => {
    await deleteHistoryItem(id)
    setHistory(prev => prev.filter(h => h._id !== id))
  }

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Download links */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Download Exports</h3>
        <div className="grid sm:grid-cols-3 gap-3">
          <DownloadBtn href={result.excel_file_url}  label="Excel (.xlsx)" icon={FileSpreadsheet} color="emerald" />
          <DownloadBtn href={result.json_output_url}  label="JSON Data"    icon={FileJson}        color="blue" />
          <DownloadBtn href={result.text_report_url}  label="TXT Report"   icon={FileText}        color="orange" />
        </div>
      </div>

      {/* Report preview */}
      {result.text_report_preview && (
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Report Preview</h3>
          <pre className="glass-light p-4 text-xs text-gray-400 font-mono whitespace-pre-wrap leading-relaxed max-h-52 overflow-y-auto">
            {result.text_report_preview}
          </pre>
        </div>
      )}

      {/* Google Sheets sync */}
      <div className="flex items-center gap-3 glass-light px-4 py-3">
        {result.gsheets_synced
          ? <><CheckCircle size={16} className="text-emerald-400" /><span className="text-sm text-emerald-400">Google Sheets synced</span></>
          : <><XCircle    size={16} className="text-gray-600"    /><span className="text-sm text-gray-500">Google Sheets not synced (credentials or Sheet ID not configured)</span></>
        }
      </div>

      {/* MongoDB DB ID */}
      {result.db_id && (
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <span className="field-label">MongoDB ID</span>
          <span className="font-mono text-gray-500">{result.db_id}</span>
        </div>
      )}

      {/* History */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-400">Past Extractions (MongoDB)</h3>
          <button onClick={loadHistory} className="btn-ghost text-xs py-1 px-3" disabled={loadingHistory}>
            {loadingHistory ? 'Loading…' : 'Load history'}
          </button>
        </div>
        {history && (
          history.length === 0
            ? <p className="text-sm text-gray-600 italic">No history found.</p>
            : <div className="space-y-1.5 max-h-64 overflow-y-auto">
                {history.map(h => <HistoryRow key={h._id} item={h} onDelete={handleDelete} />)}
              </div>
        )}
      </div>
    </div>
  )
}
