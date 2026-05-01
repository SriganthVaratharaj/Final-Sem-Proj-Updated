function RegionCard({ region }) {
  const lines = region.content_lines || []

  return (
    <div className="glass-light p-3 space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-900 capitalize">{region.section || 'section'}</span>
        <span className="text-xs text-gray-500">{lines.length} line{lines.length !== 1 ? 's' : ''}</span>
      </div>
      {region.description && <p className="text-xs text-gray-600">{region.description}</p>}
      {lines.length > 0 && (
        <div className="space-y-1 max-h-28 overflow-y-auto">
          {lines.slice(0, 8).map((line, i) => (
            <div key={i} className="text-xs text-gray-700 truncate">{line}</div>
          ))}
          {lines.length > 8 && <div className="text-xs text-gray-500">+{lines.length - 8} more lines</div>}
        </div>
      )}
    </div>
  )
}

function BlockCard({ block }) {
  return (
    <div className="glass-light p-3 space-y-1">
      <div className="text-xs font-medium uppercase text-gray-600">{(block.block_type || 'block').replace('_', ' ')}</div>
      {block.label && <div className="text-sm text-gray-800">{block.label}</div>}
      {block.content_text && <div className="text-xs text-gray-600">{block.content_text}</div>}
    </div>
  )
}

export default function LayoutTab({ result }) {
  const regions = result.layout_regions || []
  const blocks = result.detected_blocks || []
  const status = result.layoutlm_status || {}
  const embedding = result.layoutlm_embedding_preview || []

  return (
    <div className="space-y-5">
      <div className="glass-light p-4 space-y-1">
        <div className="text-sm font-medium text-gray-900">LayoutLMv3 Status</div>
        <p className="text-sm text-gray-700">{status.executed ? 'Embedding retrieved' : 'Spatial analysis only'}</p>
        <p className="text-xs text-gray-600">{status.note || 'Layout analysis complete.'}</p>
        {embedding.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-1">
            {embedding.map((v, i) => (
              <span key={i} className="tag font-mono">{v}</span>
            ))}
          </div>
        )}
      </div>

      <div className="text-sm text-gray-700">
        <span className="font-medium">Document Type:</span> <span className="capitalize">{result.document_type || 'Unknown'}</span>
      </div>

      {regions.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Layout Regions</h3>
          <div className="grid gap-2">
            {regions.map((r, i) => <RegionCard key={i} region={r} />)}
          </div>
        </div>
      )}

      {blocks.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Detected Blocks ({blocks.length})</h3>
          <div className="grid sm:grid-cols-2 gap-2">
            {blocks.map((b, i) => <BlockCard key={i} block={b} />)}
          </div>
        </div>
      )}
    </div>
  )
}
