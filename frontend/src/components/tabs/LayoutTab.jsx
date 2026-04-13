// frontend/src/components/tabs/LayoutTab.jsx
import { Layout, Layers, CheckCircle, AlertCircle } from 'lucide-react'

function RegionCard({ region }) {
  const lines = region.content_lines || []
  const sectionColors = {
    header: 'border-l-brand-500',
    body:   'border-l-emerald-600',
    footer: 'border-l-orange-500',
  }
  const cls = sectionColors[region.section] || 'border-l-gray-600'
  return (
    <div className={`glass-light p-4 border-l-4 ${cls}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm capitalize text-white">{region.section}</span>
        <span className="text-xs text-gray-500">{lines.length} line{lines.length !== 1 ? 's' : ''}</span>
      </div>
      <p className="text-xs text-gray-500 mb-2">{region.description}</p>
      {lines.length > 0 && (
        <div className="space-y-0.5 mt-2 max-h-32 overflow-y-auto">
          {lines.slice(0, 8).map((line, i) => (
            <div key={i} className="text-xs text-gray-400 truncate">{line}</div>
          ))}
          {lines.length > 8 && <div className="text-xs text-gray-600">+{lines.length - 8} more lines</div>}
        </div>
      )}
    </div>
  )
}

function BlockCard({ block }) {
  const blockColors = {
    table_region:   'bg-blue-900/20 text-blue-400 border-blue-700/40',
    totals_region:  'bg-green-900/20 text-green-400 border-green-700/40',
    key_text_block: 'bg-purple-900/20 text-purple-400 border-purple-700/40',
  }
  const cls = blockColors[block.block_type] || 'bg-gray-900/40 text-gray-400 border-gray-700/40'
  return (
    <div className={`glass-light p-3 border ${cls} rounded-xl`}>
      <div className="text-xs font-semibold uppercase tracking-wide mb-1">{block.block_type?.replace('_', ' ')}</div>
      {block.label && <div className="text-xs text-gray-500">{block.label}</div>}
      {block.content_text && (
        <div className="text-xs text-gray-400 mt-1 line-clamp-2">{block.content_text}</div>
      )}
    </div>
  )
}

export default function LayoutTab({ result }) {
  const regions = result.layout_regions  || []
  const blocks  = result.detected_blocks || []
  const status  = result.layoutlm_status || {}
  const embedding = result.layoutlm_embedding_preview || []

  return (
    <div className="space-y-6 animate-slide-up">
      {/* LayoutLMv3 status */}
      <div className="glass-light p-4 flex items-start gap-3">
        {status.executed
          ? <CheckCircle size={18} className="text-emerald-400 shrink-0 mt-0.5" />
          : <AlertCircle size={18} className="text-yellow-500 shrink-0 mt-0.5" />
        }
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-white">LayoutLMv3</span>
            <span className="tag bg-brand-900/50 text-brand-300 border border-brand-700/40">Hugging Face API</span>
            {status.executed
              ? <span className="tag bg-emerald-900/40 text-emerald-300">‣ Embedding retrieved</span>
              : <span className="tag bg-yellow-900/40 text-yellow-400">‣ Spatial analysis only</span>
            }
          </div>
          <p className="text-xs text-gray-500 mt-1">{status.note || 'Layout analysis complete.'}</p>
          {embedding.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {embedding.map((v, i) => (
                <span key={i} className="tag bg-gray-800 text-gray-400 font-mono text-xs">{v}</span>
              ))}
              <span className="text-xs text-gray-600 mt-1">CLS embedding preview</span>
            </div>
          )}
        </div>
      </div>

      {/* Document type */}
      <div className="flex items-center gap-3">
        <Layout size={16} className="text-brand-400" />
        <span className="text-sm text-gray-400">Document Type:</span>
        <span className="text-sm font-semibold text-white capitalize">{result.document_type || 'Unknown'}</span>
      </div>

      {/* Layout regions */}
      {regions.length > 0 && (
        <div>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-gray-400 mb-3">
            <Layout size={14} /> Layout Regions
          </h3>
          <div className="grid gap-3">
            {regions.map((r, i) => <RegionCard key={i} region={r} />)}
          </div>
        </div>
      )}

      {/* Detected blocks */}
      {blocks.length > 0 && (
        <div>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-gray-400 mb-3">
            <Layers size={14} /> Detected Blocks ({blocks.length})
          </h3>
          <div className="grid sm:grid-cols-2 gap-2">
            {blocks.map((b, i) => <BlockCard key={i} block={b} />)}
          </div>
        </div>
      )}
    </div>
  )
}
