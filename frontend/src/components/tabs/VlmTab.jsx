const SOURCE_META = {
  llava_local_with_ocr_hint: { label: 'Local CUDA Qwen-VL (with OCR Hint)', tier: 'GPU' },
  llava_local_only: { label: 'Local CUDA Qwen-VL (Visual Only)', tier: 'GPU' },
  image_only_fallback: { label: 'Local Fallback (Visual Only)', tier: 'GPU' },
  llava: { label: 'LLaVA 1.6 Mistral-7B', tier: 'Tier 1' },
  blip2: { label: 'BLIP-2 Flan-T5-XL', tier: 'Tier 2' },
  local: { label: 'BLIP-base (local)', tier: 'Tier 3' },
  unavailable: { label: 'Model Unavailable', tier: '' },
}

function FieldCard({ label, value }) {
  const notFound = !value || value === 'Not found' || value === 'null' || value === '—'

  return (
    <div className="glass-light p-3 flex flex-col gap-1 border-l-2 border-primary/20">
      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">{label}</span>
      <span className={`text-sm font-medium ${notFound ? 'text-gray-400 italic' : 'text-gray-900'}`}>
        {notFound ? 'Not detected' : value}
      </span>
    </div>
  )
}

export default function VlmTab({ result }) {
  const rawFields = result.vlm_fields || {}
  const templateFields = result.template_fields || {}
  const source = result.vlm_source || 'unavailable'
  const meta = SOURCE_META[source] || SOURCE_META.unavailable

  return (
    <div className="space-y-6 pb-10">
      {/* Source Info */}
      <div className="glass-light p-4 flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">VLM Inference Engine</div>
          <p className="text-sm text-gray-900 font-semibold mt-0.5">{meta.label}</p>
        </div>
        {meta.tier && <span className="tag-indigo">{meta.tier}</span>}
      </div>

      {/* Standardized Template Section */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <div className="h-4 w-1 bg-primary rounded-full"></div>
          <h3 className="text-sm font-bold text-gray-800">Standardized Layout Template</h3>
          <span className="text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded uppercase font-bold ml-auto">Recommended</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {Object.entries(templateFields).filter(([k]) => !k.startsWith('_')).map(([k, v]) => (
            <FieldCard
              key={k}
              label={k.replace(/_/g, ' ')}
              value={v}
            />
          ))}
        </div>
      </div>

      {/* Raw Extraction Section (Hidden if empty) */}
      {Object.entries(rawFields).length > 0 && (
        <div className="opacity-80 grayscale hover:grayscale-0 hover:opacity-100 transition-all">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-4 w-1 bg-gray-400 rounded-full"></div>
            <h3 className="text-sm font-bold text-gray-600">Raw Model Extraction</h3>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {Object.entries(rawFields).map(([k, v]) => (
              <FieldCard
                key={k}
                label={k.replace(/_/g, ' ')}
                value={v}
              />
            ))}
          </div>
        </div>
      )}

      <div className="glass-light p-4 text-[11px] text-gray-500 italic leading-relaxed border-t border-gray-100">
        Local Qwen-VL model is used with CUDA acceleration. The template mapper standardizes output for consistent accounting exports.
      </div>
    </div>
  )
}
