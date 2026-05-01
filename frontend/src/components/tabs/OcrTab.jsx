function LangBadge({ lang, conf }) {
  return (
    <span className="badge">
      {lang} {conf ? `· ${(conf * 100).toFixed(0)}%` : ''}
    </span>
  )
}

function FieldRow({ label, value }) {
  const notFound = !value || value === 'null' || value === 'Not found' || value === 'None'
  return (
    <div className="glass-light px-4 py-3 flex justify-between items-center gap-4">
      <span className="field-label shrink-0">{label}</span>
      <span className={`text-sm font-medium ${notFound ? 'text-gray-500 italic' : 'text-gray-900'}`}>
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
    <div className="space-y-5">
      {detected.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Languages Detected</h3>
          <div className="flex flex-wrap gap-2">
            {detected.map(d => (
              <LangBadge key={d.language} lang={d.language} conf={d.avg_confidence} />
            ))}
          </div>
        </div>
      )}

      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Extracted Fields</h3>
        <div className="space-y-2">
          {Object.entries(fields).length > 0 ? (
            Object.entries(fields).map(([k, v]) => (
              <FieldRow key={k} label={k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} value={v} />
            ))
          ) : (
            <div className="glass-light px-4 py-3 text-sm text-gray-500 italic">No structured fields found.</div>
          )}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Raw OCR Text ({texts.length} tokens)</h3>
        <div className="glass-light p-4 max-h-64 overflow-y-auto">
          <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed font-mono">
            {texts.join('\n') || 'No text extracted.'}
          </p>
        </div>
      </div>
    </div>
  )
}
