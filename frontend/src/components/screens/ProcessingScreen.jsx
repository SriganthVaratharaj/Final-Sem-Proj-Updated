// frontend/src/components/screens/ProcessingScreen.jsx
import { FileSearch, Layout, Brain, Download, CheckCircle, Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'

const STAGES = [
  { key: 'ocr',    label: 'Extracting Text (OCR)', Icon: FileSearch, color: 'brand' },
  { key: 'layout', label: 'Spatial Mapping (Layout)', Icon: Layout, color: 'emerald' },
  { key: 'vlm',    label: 'Semantic Understanding (VLM)', Icon: Brain, color: 'purple' },
  { key: 'export', label: 'Exporting Data', Icon: Download, color: 'orange' }
]

export default function ProcessingScreen({ stage, error, maxTotal=1, activeImageNumber=1 }) {
  // Map backend stages to indices for stepper
  const getStageIndex = (s) => STAGES.findIndex(st => st.key === s)
  const currentIndex = getStageIndex(stage)

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6 animate-fade-in p-6 glass rounded-3xl relative overflow-hidden">
      {/* Background Pulse Effect */}
      <div className="absolute -top-40 -left-40 w-96 h-96 bg-brand-600/20 rounded-full blur-[100px] animate-pulse"></div>

      <div className="relative z-10 text-center mb-8">
        <h2 className="text-2xl font-bold text-white">AI Processing</h2>
        <p className="text-sm text-gray-400 mt-2">
          Image {activeImageNumber} of {maxTotal} running through multi-modal pipeline
        </p>
      </div>

      <div className="relative z-10 space-y-4">
        {STAGES.map((s, idx) => {
          const isActive = idx === currentIndex
          const isDone = currentIndex > idx || (stage === 'export' && currentIndex !== -1) // Assuming export means done.
          const isPending = currentIndex < idx && currentIndex !== -1

          const statusColors = {
            active: `border-${s.color}-500 bg-${s.color}-900/20 text-${s.color}-300 ring-4 ring-${s.color}-900/30`,
            done: `border-emerald-500/50 bg-emerald-900/10 text-emerald-400`,
            pending: `border-gray-800 bg-gray-900/50 text-gray-600`
          }

          const currentStyle = isDone ? statusColors.done : (isActive ? statusColors.active : statusColors.pending)

          return (
            <div key={s.key} className={`flex items-center gap-4 p-4 rounded-2xl border transition-all duration-500 ${currentStyle} ${isActive ? 'scale-[1.02]' : ''}`}>
              <div className={`p-2 rounded-xl ${isActive ? `bg-${s.color}-900/50` : (isDone ? 'bg-emerald-900/50' : 'bg-gray-800')}`}>
                <s.Icon size={20} className={isActive ? 'animate-bounce' : ''} />
              </div>
              
              <div className="flex-1 font-semibold text-lg flex items-center justify-between">
                <span>{s.label}</span>
                {isActive && <Loader2 size={18} className="animate-spin opacity-70" />}
                {isDone && <CheckCircle size={18} className="text-emerald-500" />}
              </div>
            </div>
          )
        })}
      </div>

      {error && (
        <div className="relative z-10 mt-6 p-4 rounded-xl border border-red-900/50 bg-red-900/20 text-red-400 text-sm font-semibold flex items-center gap-2">
          <Loader2 className="animate-spin opacity-0 shrink-0" size={16}/> { /* invisible spacer */ }
          {error}
        </div>
      )}
    </div>
  )
}
