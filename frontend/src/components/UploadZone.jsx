// frontend/src/components/UploadZone.jsx
import { Upload, X, Image as ImageIcon } from 'lucide-react'
import { useCallback, useState } from 'react'

export default function UploadZone({ onSubmit, disabled }) {
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)

  const accept = '.jpg,.jpeg,.png,.bmp,.tif,.tiff'

  const addFiles = useCallback((fileList) => {
    const allowed = Array.from(fileList).filter(f => /\.(jpe?g|png|bmp|tiff?)$/i.test(f.name))
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...allowed.filter(f => !names.has(f.name))]
    })
  }, [])

  const removeFile = (name) => setFiles(prev => prev.filter(f => f.name !== name))
  const clearAll   = () => setFiles([])

  const onDrop = (e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }
  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onChange = (e) => { addFiles(e.target.files); e.target.value = '' }

  const handleSubmit = () => { if (files.length) { onSubmit(files); setFiles([]) } }

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <label
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`flex flex-col items-center justify-center gap-4 p-10 rounded-2xl border-2 border-dashed
          cursor-pointer transition-all duration-300 group
          ${dragging
            ? 'border-brand-400 bg-brand-900/20 scale-[1.01]'
            : 'border-gray-700 hover:border-brand-600 hover:bg-brand-900/10 bg-gray-900/30'
          }
          ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      >
        <div className={`p-4 rounded-full bg-brand-900/50 border border-brand-700/50 transition-transform group-hover:scale-110 ${dragging ? 'scale-110' : ''}`}>
          <Upload size={28} className="text-brand-400" />
        </div>
        <div className="text-center">
          <p className="text-white font-semibold text-lg">Drop invoices here, or click to browse</p>
          <p className="text-gray-500 text-sm mt-1">JPG, PNG, BMP, TIFF — up to 10 MB each</p>
        </div>
        <input type="file" multiple accept={accept} onChange={onChange} className="hidden" disabled={disabled} />
      </label>

      {/* File list */}
      {files.length > 0 && (
        <div className="glass p-4 space-y-2 animate-fade-in">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-gray-300">{files.length} file{files.length > 1 ? 's' : ''} selected</span>
            <button onClick={clearAll} className="text-xs text-gray-500 hover:text-red-400 transition-colors">Clear all</button>
          </div>
          {files.map(f => (
            <div key={f.name} className="flex items-center gap-3 glass-light px-3 py-2">
              <ImageIcon size={16} className="text-brand-400 shrink-0" />
              <span className="text-sm text-gray-300 truncate flex-1">{f.name}</span>
              <span className="text-xs text-gray-600 shrink-0">{(f.size / 1024).toFixed(0)} KB</span>
              <button onClick={() => removeFile(f.name)} className="text-gray-600 hover:text-red-400 transition-colors shrink-0">
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={!files.length || disabled}
        className="btn-primary w-full flex items-center justify-center gap-2"
      >
        <Upload size={18} />
        Process {files.length > 0 ? `${files.length} Image${files.length > 1 ? 's' : ''}` : 'Images'}
      </button>
    </div>
  )
}
