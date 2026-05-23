import React from 'react'

export default function DigitalTwinTab({ result }) {
  const content = result.digital_twin_content || ""

  if (!content) {
    return (
      <div className="glass-light p-8 text-center">
        <p className="text-sm text-gray-500 italic">Spatial reconstruction failed for this image.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-gray-800">Digital Twin (Spatial Reconstruction)</h3>
          <p className="text-[11px] text-gray-500 mt-0.5">
            Preserves the original bill format. Best viewed on desktop.
          </p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 overflow-x-auto shadow-inner">
        <pre className="text-[10px] md:text-xs leading-[1.1] font-mono text-gray-800 whitespace-pre select-all">
          {content}
        </pre>
      </div>

      <div className="glass-light p-4 text-[11px] text-gray-600 leading-relaxed border-t border-gray-100 flex gap-3 items-start">
        <span className="text-lg">💡</span>
        <span>
          <strong>Pro-tip:</strong> This view converts stylized/handwritten fonts into standard digital text while keeping the original positions (Digital Twin). You can copy-paste this directly into your ledger or accounting software.
        </span>
      </div>
    </div>
  )
}
