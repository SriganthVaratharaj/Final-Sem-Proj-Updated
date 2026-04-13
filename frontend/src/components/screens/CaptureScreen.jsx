// frontend/src/components/screens/CaptureScreen.jsx
import { Upload, Camera, Sparkles } from 'lucide-react'
import { useCallback, useState } from 'react'

export default function CaptureScreen({ onSubmit, disabled }) {
  const [dragging, setDragging] = useState(false)
  const accept = '.jpg,.jpeg,.png,.bmp,.tif,.tiff,.pdf'

  const handleFiles = useCallback((fileList) => {
    const allowed = Array.from(fileList).filter(f => /\.(jpe?g|png|bmp|tiff?|pdf)$/i.test(f.name))
    if (allowed.length > 0) {
      onSubmit(allowed)
    }
  }, [onSubmit])

  const onDrop = (e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }
  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onChange = (e) => { handleFiles(e.target.files); e.target.value = '' }

  return (
    <div className="space-y-6 animate-fade-in w-full max-w-2xl mx-auto">
      <section className="text-center space-y-3 mb-8">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-brand-900/50 border border-brand-700/50 text-brand-300 text-xs font-semibold mb-2">
          <Sparkles size={12} /> Combined AI Pipeline
        </div>
        <h1 className="text-3xl md:text-5xl font-extrabold text-white leading-tight">
          Scan or Upload
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-violet-400">
            Invoice
          </span>
        </h1>
        <p className="text-gray-400 text-sm leading-relaxed max-w-md mx-auto">
          Take a photo on your mobile device, or upload directly from your computer.
        </p>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Mobile Camera Button (capture="environment") */}
        <label
          className={`flex flex-col items-center justify-center gap-3 p-8 rounded-2xl border-2 border-brand-700 bg-brand-900/20 cursor-pointer hover:bg-brand-900/30 transition-all ${disabled ? 'pointer-events-none opacity-50' : ''}`}
        >
          <div className="p-4 rounded-full bg-brand-900 border border-brand-500">
            <Camera size={32} className="text-brand-400" />
          </div>
          <div className="text-center">
            <p className="text-white font-bold text-lg">Use Camera</p>
            <p className="text-gray-400 text-xs mt-1">Scan via rear camera</p>
          </div>
          {/* Magic attribute for mobile camera */}
          <input type="file" accept="image/*" capture="environment" onChange={onChange} className="hidden" disabled={disabled} />
        </label>

        {/* Standard Upload / Dropzone */}
        <label
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`flex flex-col items-center justify-center gap-3 p-8 rounded-2xl border-2 border-dashed
            cursor-pointer transition-all duration-300 group
            ${dragging ? 'border-violet-400 bg-violet-900/20 scale-[1.02]' : 'border-gray-700 bg-gray-900/30 hover:border-violet-600 hover:bg-violet-900/10'}
            ${disabled ? 'pointer-events-none opacity-50' : ''}`}
        >
          <div className={`p-4 rounded-full bg-gray-800 border border-gray-600 transition-transform group-hover:scale-110 ${dragging ? 'scale-110 border-violet-500' : ''}`}>
            <Upload size={32} className={dragging ? 'text-violet-400' : 'text-gray-400'} />
          </div>
          <div className="text-center">
            <p className="text-white font-bold text-lg">Upload File</p>
            <p className="text-gray-400 text-xs mt-1">Images or PDFs under 10MB</p>
          </div>
          <input type="file" multiple accept={accept} onChange={onChange} className="hidden" disabled={disabled} />
        </label>
      </div>
    </div>
  )
}
