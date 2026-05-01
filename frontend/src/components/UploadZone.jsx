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
  const clearAll = () => setFiles([])

  const onDrop = (e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }
  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onChange = (e) => { addFiles(e.target.files); e.target.value = '' }

  const handleSubmit = () => { if (files.length) { onSubmit(files); setFiles([]) } }

  return (
    <div className="space-y-4">
      <label
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`flex flex-col items-center justify-center gap-2 p-8 rounded-md border-2 border-dashed cursor-pointer transition-colors ${
          dragging ? 'border-gray-500 bg-gray-100' : 'border-gray-300 bg-white hover:bg-gray-50'
        } ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      >
        <div className="text-center">
          <p className="text-gray-900 font-medium text-base">Drop invoice images here or click to browse</p>
          <p className="text-gray-600 text-sm mt-1">JPG, PNG, BMP, TIFF up to 10 MB each</p>
        </div>
        <input type="file" multiple accept={accept} onChange={onChange} className="hidden" disabled={disabled} />
      </label>

      {files.length > 0 && (
        <div className="glass p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">
              {files.length} file{files.length > 1 ? 's' : ''} selected
            </span>
            <button onClick={clearAll} className="btn-ghost text-xs py-1 px-2">Clear all</button>
          </div>
          {files.map(f => (
            <div key={f.name} className="glass-light px-3 py-2 flex items-center gap-2">
              <span className="text-sm text-gray-800 truncate flex-1">{f.name}</span>
              <span className="text-xs text-gray-600">{(f.size / 1024).toFixed(0)} KB</span>
              <button onClick={() => removeFile(f.name)} className="btn-ghost text-xs py-1 px-2">Remove</button>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={!files.length || disabled}
        className="btn-primary w-full"
      >
        Process {files.length > 0 ? `${files.length} Image${files.length > 1 ? 's' : ''}` : 'Images'}
      </button>
    </div>
  )
}
