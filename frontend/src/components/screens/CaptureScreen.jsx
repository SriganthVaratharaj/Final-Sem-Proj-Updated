import { useCallback, useState } from 'react'

export default function CaptureScreen({ onSubmit, disabled, error }) {
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
    <div className="space-y-6 w-full max-w-3xl mx-auto">
      <section className="text-center space-y-2">
        <h1 className="text-3xl md:text-4xl font-semibold text-gray-900">Upload Invoice</h1>
        <p className="text-gray-600 text-sm">
          Use camera or upload files to start extraction.
        </p>
      </section>

      {error && (
        <div className="p-3 rounded-md border border-red-200 bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <label
          className={`flex flex-col items-center justify-center gap-2 p-6 rounded-md border border-gray-300 bg-white cursor-pointer hover:bg-gray-50 ${disabled ? 'pointer-events-none opacity-50' : ''}`}
        >
          <div className="text-center">
            <p className="text-gray-900 font-medium">Use Camera</p>
            <p className="text-gray-600 text-sm">Open mobile rear camera</p>
          </div>
          <input type="file" accept="image/*" capture="environment" onChange={onChange} className="hidden" disabled={disabled} />
        </label>

        <label
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`flex flex-col items-center justify-center gap-2 p-6 rounded-md border-2 border-dashed
            cursor-pointer transition-colors
            ${dragging ? 'border-gray-500 bg-gray-100' : 'border-gray-300 bg-white hover:bg-gray-50'}
            ${disabled ? 'pointer-events-none opacity-50' : ''}`}
        >
          <div className="text-center">
            <p className="text-gray-900 font-medium">Upload File</p>
            <p className="text-gray-600 text-sm">Images or PDFs under 10MB</p>
          </div>
          <input type="file" multiple accept={accept} onChange={onChange} className="hidden" disabled={disabled} />
        </label>
      </div>
    </div>
  )
}
