// frontend/src/components/tabs/VlmTab.jsx
import { Brain, Cpu, Cloud, Server } from 'lucide-react'

const SOURCE_META = {
  llava: { label: 'LLaVA 1.6 Mistral-7B',  icon: Cloud,  cls: 'bg-brand-900/40 text-brand-300 border-brand-700/40', tier: 'Tier 1' },
  blip2: { label: 'BLIP-2 Flan-T5-XL',      icon: Cloud,  cls: 'bg-purple-900/40 text-purple-300 border-purple-700/40', tier: 'Tier 2' },
  local: { label: 'BLIP-base (local)',        icon: Cpu,    cls: 'bg-orange-900/40 text-orange-300 border-orange-700/40', tier: 'Tier 3' },
  unavailable: { label: 'Unavailable',        icon: Server, cls: 'bg-gray-900 text-gray-500 border-gray-700', tier: '' },
}

function FieldCard({ label, value }) {
  const notFound = !value || value === 'Not found' || value === 'null'
  return (
    <div className="glass-light p-4 flex flex-col gap-1">
      <span className="field-label">{label}</span>
      <span className={`text-base font-semibold ${notFound ? 'text-gray-600 italic text-sm' : 'text-white'}`}>
        {notFound ? 'Not detected' : value}
      </span>
    </div>
  )
}

export default function VlmTab({ result }) {
  const fields = result.vlm_fields  || {}
  const source = result.vlm_source  || 'unavailable'
  const meta   = SOURCE_META[source] || SOURCE_META.unavailable
  const Icon   = meta.icon

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Model info */}
      <div className={`glass-light p-4 flex items-center gap-3 border ${meta.cls}`}>
        <div className={`p-2 rounded-lg border ${meta.cls}`}>
          <Icon size={16} />
        </div>
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-white">
              <Brain size={14} className="inline mr-1 text-brand-400" />VLM Source
            </span>
            {meta.tier && <span className={`tag border ${meta.cls}`}>{meta.tier}</span>}
          </div>
          <p className="text-xs text-gray-500 mt-0.5">{meta.label}</p>
        </div>
      </div>

      {/* Extracted fields 2-col grid */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Extracted Invoice Fields</h3>
        <div className="grid sm:grid-cols-2 gap-2">
          <FieldCard label="Vendor / Company Name" value={fields.vendor_name} />
          <FieldCard label="Invoice Number"         value={fields.invoice_number} />
          <FieldCard label="Invoice Date"           value={fields.date} />
          <FieldCard label="Total Amount"           value={fields.total_amount} />
        </div>
      </div>

      {/* Description */}
      <div className="glass-light p-4 text-xs text-gray-500 leading-relaxed">
        <strong className="text-gray-400">How it works: </strong>
        The image is sent to LLaVA 1.6 (Tier 1) with a structured prompt.
        If the API call fails, BLIP-2 (Tier 2) is tried next.
        Finally, a local BLIP-base model runs iterative per-field prompts as the last resort.
        All API calls use your <code className="text-brand-400">HF_TOKEN</code>.
      </div>
    </div>
  )
}
