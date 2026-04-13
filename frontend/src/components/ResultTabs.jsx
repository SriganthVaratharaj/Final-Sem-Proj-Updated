// frontend/src/components/ResultTabs.jsx
import { useState } from 'react'
import { FileSearch, Layout, Brain, Download, CheckCircle, XCircle } from 'lucide-react'
import OcrTab    from './tabs/OcrTab'
import LayoutTab from './tabs/LayoutTab'
import VlmTab    from './tabs/VlmTab'
import ExportsTab from './tabs/ExportsTab'

const TABS = [
  { key: 'ocr',    label: 'OCR',    Icon: FileSearch, color: 'brand' },
  { key: 'layout', label: 'Layout', Icon: Layout,     color: 'emerald' },
  { key: 'vlm',    label: 'VLM',    Icon: Brain,      color: 'purple' },
  { key: 'exports',label: 'Exports',Icon: Download,   color: 'orange' },
]

function ResultCard({ result, defaultOpen }) {
  const [active, setActive] = useState('ocr')
  const [expanded, setExpanded] = useState(defaultOpen)

  const success = result.status === 'success'

  return (
    <div className="glass animate-slide-up overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-5 text-left hover:bg-white/5 transition-colors"
      >
        {success
          ? <CheckCircle size={18} className="text-emerald-400 shrink-0" />
          : <XCircle     size={18} className="text-red-400 shrink-0" />
        }
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-white truncate">{result.image_name}</div>
          {result.error && <div className="text-xs text-red-400 mt-0.5">{result.error}</div>}
          {success && (
            <div className="flex flex-wrap gap-1.5 mt-1.5">
              <span className="tag bg-gray-800 text-gray-400 capitalize">{result.document_type}</span>
              <span className="tag bg-brand-900/50 text-brand-300">{result.ocr_texts?.length || 0} words</span>
              {result.vlm_source && result.vlm_source !== 'unavailable' && (
                <span className="tag bg-purple-900/40 text-purple-300">VLM: {result.vlm_source}</span>
              )}
            </div>
          )}
        </div>
        <span className="text-gray-600 text-xs shrink-0">{expanded ? '▲' : '▼'}</span>
      </button>

      {/* Tabs + content */}
      {expanded && success && (
        <div className="border-t border-gray-700/50">
          {/* Tab bar */}
          <div className="flex border-b border-gray-800">
            {TABS.map(({ key, label, Icon, color }) => (
              <button
                key={key}
                onClick={() => setActive(key)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-semibold transition-all duration-200
                  ${active === key
                    ? `border-b-2 border-${color}-500 text-${color}-300 bg-${color}-900/10`
                    : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                  }`}
              >
                <Icon size={13} /> {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="p-5">
            {active === 'ocr'     && <OcrTab     result={result} />}
            {active === 'layout'  && <LayoutTab  result={result} />}
            {active === 'vlm'     && <VlmTab     result={result} />}
            {active === 'exports' && <ExportsTab result={result} />}
          </div>
        </div>
      )}
    </div>
  )
}

export default function ResultTabs({ results }) {
  if (!results || results.length === 0) return null
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-bold text-white">Results ({results.length})</h2>
      {results.map((r, i) => (
        <ResultCard key={i} result={r} defaultOpen={i === 0} />
      ))}
    </div>
  )
}
