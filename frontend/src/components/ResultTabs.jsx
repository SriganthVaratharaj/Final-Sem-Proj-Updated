import { useState } from 'react'
import VlmTab from './tabs/VlmTab'
import DigitalTwinTab from './tabs/DigitalTwinTab'
import ExportsTab from './tabs/ExportsTab'

const TABS = [
  { key: 'vlm', label: 'Structured Extraction' },
  { key: 'digital_twin', label: 'Spatial Twin' },
  { key: 'exports', label: 'Exports' },
]

function ResultCard({ result, defaultOpen }) {
  const [active, setActive] = useState('vlm')
  const [expanded, setExpanded] = useState(defaultOpen)

  const success = result.status === 'success'

  return (
    <div className="border border-gray-200 rounded-md overflow-hidden bg-white">
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
        <div className="border-t border-gray-200">
          <div className="flex border-b border-gray-200 overflow-x-auto">
            {TABS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActive(key)}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                  active === key
                    ? 'border-gray-900 text-gray-900'
                    : 'border-transparent text-gray-500 hover:text-gray-800 hover:bg-gray-50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="p-4">
            {active === 'vlm' && <VlmTab result={result} />}
            {active === 'digital_twin' && <DigitalTwinTab result={result} />}
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
      <h2 className="text-lg font-semibold text-gray-900">Results ({results.length})</h2>
      {results.map((r, i) => (
        <ResultCard key={i} result={r} defaultOpen={i === 0} />
      ))}
    </div>
  )
}
