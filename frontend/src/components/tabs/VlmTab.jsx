const SOURCE_META = {
  llava: { label: 'LLaVA 1.6 Mistral-7B', tier: 'Tier 1' },
  blip2: { label: 'BLIP-2 Flan-T5-XL', tier: 'Tier 2' },
  local: { label: 'BLIP-base (local)', tier: 'Tier 3' },
  unavailable: { label: 'Unavailable', tier: '' },
}

function FieldCard({ label, value }) {
  const notFound = !value || value === 'Not found' || value === 'null'

  return (
    <div className="glass-light p-3 flex flex-col gap-1">
      <span className="field-label">{label}</span>
      <span className={`text-sm font-medium ${notFound ? 'text-gray-500 italic' : 'text-gray-900'}`}>
        {notFound ? 'Not detected' : value}
      </span>
    </div>
  )
}

export default function VlmTab({ result }) {
  const fields = result.vlm_fields || {}
  const source = result.vlm_source || 'unavailable'
  const meta = SOURCE_META[source] || SOURCE_META.unavailable

  return (
    <div className="space-y-5">
      <div className="glass-light p-4">
        <div className="text-sm font-medium text-gray-900">VLM Source</div>
        <p className="text-sm text-gray-700 mt-1">{meta.label}</p>
        {meta.tier && <span className="tag mt-2">{meta.tier}</span>}
      </div>

      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Extracted Invoice Fields</h3>
        <div className="grid sm:grid-cols-2 gap-2">
          {Object.entries(fields).length > 0 ? (
            Object.entries(fields).map(([k, v]) => (
              <FieldCard
                key={k}
                label={k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                value={v}
              />
            ))
          ) : (
            <div className="sm:col-span-2 glass-light p-4 text-sm text-gray-500 italic">
              No structured fields detected by VLM.
            </div>
          )}
        </div>
      </div>

      <div className="glass-light p-4 text-xs text-gray-600 leading-relaxed">
        LLaVA is used first. If unavailable, BLIP-2 is tried. If both fail, local BLIP-base fallback runs.
      </div>
    </div>
  )
}
