// frontend/src/components/ProgressBar.jsx
import { CheckCircle, Loader2, Zap } from 'lucide-react'

const STAGES = [
  { key: 'ocr',    label: 'PaddleOCR',   desc: 'Multi-language text extraction' },
  { key: 'layout', label: 'LayoutLMv3',  desc: 'Spatial layout analysis (HF API)' },
  { key: 'vlm',    label: 'LLaVA VLM',   desc: 'Invoice field extraction (HF API)' },
  { key: 'export', label: 'Exporting',   desc: 'JSON + Excel + Report' },
]

function StageIcon({ status }) {
  if (status === 'done')    return <CheckCircle size={16} className="text-emerald-400" />
  if (status === 'running') return <Loader2 size={16} className="text-brand-400 animate-spin" />
  return <div className="w-4 h-4 rounded-full border-2 border-gray-700" />
}

export default function ProgressBar({ stages, images = 1 }) {
  // stages: { [imgIdx]: currentStageName }
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
    <div className="glass p-6 animate-fade-in">
      <div className="flex items-center gap-2 mb-5">
        <Zap size={16} className="text-brand-400" />
        <span className="text-sm font-semibold text-gray-300">
          Processing {images} image{images > 1 ? 's' : ''}…
        </span>
      </div>

      {/* Stage steps */}
      <div className="flex items-start gap-0">
        {STAGES.map((stage, idx) => {
          const status = getStatus(stage.key)
          const isLast = idx === STAGES.length - 1
          return (
            <div key={stage.key} className="flex items-start flex-1">
              <div className="flex flex-col items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 transition-all duration-400
                  ${status === 'done'    ? 'border-emerald-500 bg-emerald-900/30' : ''}
                  ${status === 'running' ? 'border-brand-500 bg-brand-900/40 shadow-md shadow-brand-900' : ''}
                  ${status === 'pending' ? 'border-gray-700 bg-gray-900/40' : ''}
                `}>
                  <StageIcon status={status} />
                </div>
                <div className="mt-2 text-center">
                  <div className={`text-xs font-semibold leading-tight
                    ${status === 'done'    ? 'text-emerald-400' : ''}
                    ${status === 'running' ? 'text-brand-300'   : ''}
                    ${status === 'pending' ? 'text-gray-600'    : ''}
                  `}>{stage.label}</div>
                  <div className="text-xs text-gray-600 mt-0.5 max-w-[90px] leading-tight hidden sm:block">{stage.desc}</div>
                </div>
              </div>
              {!isLast && (
                <div className={`flex-1 h-0.5 mt-4 mx-1 transition-all duration-500
                  ${status === 'done' ? 'bg-emerald-600' : 'bg-gray-700'}`} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
