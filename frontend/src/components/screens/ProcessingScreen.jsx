const STAGES = [
  { key: 'ocr', label: 'Extracting Text (OCR)' },
  { key: 'layout', label: 'Layout Analysis' },
  { key: 'vlm', label: 'Field Extraction (VLM)' },
  { key: 'export', label: 'Exporting Output' },
]

export default function ProcessingScreen({ stage, error, maxTotal = 1, activeImageNumber = 1 }) {
  const getStageIndex = (s) => STAGES.findIndex(st => st.key === s)
  const currentIndex = getStageIndex(stage)

  return (
    <div className="w-full max-w-3xl mx-auto glass p-6 space-y-5">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-gray-900">Processing</h2>
        <p className="text-sm text-gray-600 mt-1">
          Image {activeImageNumber} of {maxTotal}
        </p>
      </div>

      <div className="space-y-3">
        {STAGES.map((s, idx) => {
          const isActive = idx === currentIndex
          const isDone = currentIndex > idx || (stage === 'export' && currentIndex !== -1)

          let rowClass = 'border-gray-200 bg-white text-gray-700'
          if (isDone) rowClass = 'border-green-200 bg-green-50 text-green-700'
          if (isActive) rowClass = 'border-gray-400 bg-gray-100 text-gray-900'

          let statusText = 'Pending'
          if (isDone) statusText = 'Done'
          if (isActive) statusText = 'Running'

          return (
            <div key={s.key} className={`border rounded-md px-4 py-3 flex items-center justify-between ${rowClass}`}>
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium">{idx + 1}.</span>
                <span className="text-sm font-medium">{s.label}</span>
              </div>
              <span className="text-xs font-medium">{statusText}</span>
            </div>
          )
        })}
      </div>

      {error && (
        <div className="p-3 rounded-md border border-red-200 bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}
    </div>
  )
}
