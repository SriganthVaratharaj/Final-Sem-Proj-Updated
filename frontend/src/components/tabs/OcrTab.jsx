// frontend/src/components/tabs/OcrTab.jsx
import { Globe, BarChart2 } from 'lucide-react'

function LangBadge({ lang, conf }) {
  const colors = {
    Tamil:   'bg-orange-900/40 text-orange-300 border-orange-700/50',
    Telugu:  'bg-teal-900/40  text-teal-300  border-teal-700/50',
    'Hindi/Marathi (Devanagari)': 'bg-purple-900/40 text-purple-300 border-purple-700/50',
    Kannada: 'bg-rose-900/40  text-rose-300  border-rose-700/50',
    Bengali: 'bg-cyan-900/40  text-cyan-300  border-cyan-700/50',
    default: 'bg-brand-900/40 text-brand-300 border-brand-700/50',
  }
  const cls = colors[lang] || colors.default
  return (
    <span className={`badge border ${cls}`}>
      {lang} {conf ? `· ${(conf * 100).toFixed(0)}%` : ''}
    </span>
  )
}

function FieldRow({ label, value }) {
  const notFound = !value || value === 'null' || value === 'Not found' || value === 'None'
  return (
    <div className="glass-light px-4 py-3 flex justify-between items-center gap-4">
      <span className="field-label shrink-0">{label}</span>
      <span className={`text-sm font-medium ${notFound ? 'text-gray-600 italic' : 'text-white'}`}>
        {notFound ? '—' : String(value)}
      </span>
    </div>
  )
}

export default function OcrTab({ result }) {
  const texts  = result.ocr_texts || []
  const lang   = result.ocr_language_summary || {}
  const fields = result.ocr_invoice_fields   || {}
  const detected = lang.detected || []

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Language summary */}
      {detected.length > 0 && (
        <div>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-gray-400 mb-3">
            <Globe size={14} /> Languages Detected
          </h3>
          <div className="flex flex-wrap gap-2">
            {detected.map(d => (
              <LangBadge key={d.language} lang={d.language} conf={d.avg_confidence} />
            ))}
          </div>
        </div>
      )}

      {/* Invoice fields */}
      <div>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-gray-400 mb-3">
          <BarChart2 size={14} /> Extracted Fields (OCR Rules)
        </h3>
        <div className="space-y-1.5">
          <FieldRow label="Invoice No" value={fields.invoice_no} />
          <FieldRow label="Date" value={fields.date} />
          <FieldRow label="Phone" value={fields.phone} />
          <FieldRow label="Sub-total" value={fields.sub_total} />
          <FieldRow label="Tax" value={fields.tax} />
          <FieldRow label="Grand Total" value={fields.grand_total} />
        </div>
      </div>

      {/* Raw text */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Raw OCR Text ({texts.length} tokens)</h3>
        <div className="glass-light p-4 max-h-64 overflow-y-auto">
          <p className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed font-mono">
            {texts.join('\n') || 'No text extracted.'}
          </p>
        </div>
      </div>
    </div>
  )
}
