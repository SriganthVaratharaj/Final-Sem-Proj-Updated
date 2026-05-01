const STAGES = [
  { key: 'ocr', label: 'PaddleOCR', desc: 'Text extraction' },
  { key: 'layout', label: 'LayoutLMv3', desc: 'Layout analysis' },
  { key: 'vlm', label: 'LLaVA VLM', desc: 'Field extraction' },
  { key: 'export', label: 'Exporting', desc: 'Output generation' },
]

export default function ProgressBar({ stages, images = 1 }) {
  const stageKeys = Object.values(stages)
  const currentStage = stageKeys.length > 0 ? stageKeys[stageKeys.length - 1] : null
  const currentStageIdx = STAGES.findIndex(s => s.key === currentStage)

  function getStatus(stageKey) {
    const idx = STAGES.findIndex(s => s.key === stageKey)
    if (idx < currentStageIdx) return 'done'
    if (idx === currentStageIdx) return 'running'
    return 'pending'
  }

  return (
    <div className="glass p-4 space-y-4">
      <div className="text-sm font-medium text-gray-700">
        Processing {images} image{images > 1 ? 's' : ''}
      </div>

      <div className="grid sm:grid-cols-4 gap-2">
        {STAGES.map((stage) => {
          const status = getStatus(stage.key)

          return (
            <div
              key={stage.key}
              className={`border rounded-md p-2.5 ${
                status === 'done'
                  ? 'border-green-200 bg-green-50'
                  : status === 'running'
                    ? 'border-gray-400 bg-gray-100'
                    : 'border-gray-200 bg-white'
              }`}
            >
              <div className="text-sm font-medium text-gray-900">{stage.label}</div>
              <div className="text-xs text-gray-600">{stage.desc}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
