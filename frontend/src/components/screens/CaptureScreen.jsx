import { useCallback, useState, useRef, useEffect } from 'react'

export default function CaptureScreen({ onSubmit, disabled, error }) {
  const [dragging, setDragging] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  
  // --- Cropper State ---
  const [cropImageSrc, setCropImageSrc] = useState(null)
  const [cropRect, setCropRect] = useState({ x: 0, y: 0, w: 0, h: 0 })
  const [isDrawing, setIsDrawing] = useState(false)
  const [dragStart, setDragStart] = useState(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const imageDims = useRef({ w: 0, h: 0 })

  const accept = '.jpg,.jpeg,.png,.bmp,.tif,.tiff,.pdf'
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null) // Native camera fallback input
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  
  const containerRef = useRef(null) // Cropper image container ref
  const imgRef = useRef(null) // Cropper image element ref

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // File loading helper for cropper
  const loadFileForCropping = (file) => {
    if (!file) return
    // PDFs can't be easily cropped in canvas; bypass cropper for PDFs
    if (file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf') {
      onSubmit([file])
      return
    }
    
    const reader = new FileReader()
    reader.onload = (e) => {
      setCropImageSrc(e.target.result)
      setImageLoaded(false)
      imageDims.current = { w: 0, h: 0 }
      setCropRect({ x: 0, y: 0, w: 0, h: 0 })
      setDragStart(null)
      setIsDrawing(false)
    }
    reader.readAsDataURL(file)
  }

  const handleFiles = useCallback((fileList) => {
    if (!fileList || fileList.length === 0) return
    loadFileForCropping(fileList[0])
  }, [])

  const onDrop = (e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }
  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onChange = (e) => { handleFiles(e.target.files); e.target.value = '' }

  const handleFileClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  // --- Live Camera Scanner Methods ---
  const startCamera = async () => {
    if (disabled) return
    try {
      setCameraError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: { ideal: 'environment' }, 
          width: { ideal: 1920 }, 
          height: { ideal: 1080 } 
        },
        audio: false
      })
      
      streamRef.current = stream
      setShowCamera(true)
      
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      }, 100)
    } catch (err) {
      console.error("Camera access error:", err)
      setCameraError("Live camera access is restricted. Falling back to native device camera...")
      setTimeout(() => {
        if (cameraInputRef.current) {
          cameraInputRef.current.click()
        }
      }, 1000)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setShowCamera(false)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    if (!video || !streamRef.current) return

    const canvas = document.createElement('canvas')
    const videoWidth = video.videoWidth || 1280
    const videoHeight = video.videoHeight || 720
    
    canvas.width = videoWidth
    canvas.height = videoHeight

    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight)

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' })
        stopCamera()
        loadFileForCropping(file)
      }
    }, 'image/jpeg', 0.90) // Capture high resolution frame for cropper
  }

  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target
    imageDims.current = { w: naturalWidth || e.target.width || 800, h: naturalHeight || e.target.height || 600 }
    setImageLoaded(true)
  }

  // --- Drag Selection Cropping Event Handlers ---
  const handleDragStart = (e) => {
    if (disabled || !imageLoaded) return
    const container = containerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    
    const touch = e.touches && e.touches.length > 0 ? e.touches[0] : null
    const clientX = touch ? touch.clientX : e.clientX
    const clientY = touch ? touch.clientY : e.clientY
    
    const x = clientX - rect.left
    const y = clientY - rect.top
    
    setDragStart({ x, y })
    setCropRect({ x, y, w: 0, h: 0 })
    setIsDrawing(true)
  }

  const handleDragMove = (e) => {
    if (!isDrawing || !dragStart) return
    const container = containerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    
    const touch = e.touches && e.touches.length > 0 ? e.touches[0] : null
    const clientX = touch ? touch.clientX : e.clientX
    const clientY = touch ? touch.clientY : e.clientY
    
    const currentX = Math.max(0, Math.min(rect.width, clientX - rect.left))
    const currentY = Math.max(0, Math.min(rect.height, clientY - rect.top))
    
    const x = Math.min(dragStart.x, currentX)
    const y = Math.min(dragStart.y, currentY)
    const w = Math.abs(dragStart.x - currentX)
    const h = Math.abs(dragStart.y - currentY)
    
    setCropRect({ x, y, w, h })
  }

  const handleDragEnd = () => {
    setIsDrawing(false)
  }

  const resetCrop = () => {
    setCropRect({ x: 0, y: 0, w: 0, h: 0 })
    setDragStart(null)
  }

  const handleCropConfirm = () => {
    const img = imgRef.current
    if (!img || !imageLoaded) return

    const canvas = document.createElement('canvas')
    const naturalWidth = imageDims.current.w || img.naturalWidth || img.clientWidth || 800
    const naturalHeight = imageDims.current.h || img.naturalHeight || img.clientHeight || 600
    
    const layoutW = img.clientWidth || img.width || 800
    const layoutH = img.clientHeight || img.height || 600

    // Scale crop selection coordinates to image source size
    const scaleX = naturalWidth / layoutW
    const scaleY = naturalHeight / layoutH
    
    const cropX = cropRect.x * scaleX
    const cropY = cropRect.y * scaleY
    const cropW = cropRect.w * scaleX
    const cropH = cropRect.h * scaleY

    if (cropW < 5 || cropH < 5) return

    // Client-side downscaling cap (max 1200px width/height) to avoid Kaggle VRAM OOM
    const maxDim = 1200
    let canvasW = cropW
    let canvasH = cropH
    if (Math.max(canvasW, canvasH) > maxDim) {
      const scale = maxDim / Math.max(canvasW, canvasH)
      canvasW = Math.round(canvasW * scale)
      canvasH = Math.round(canvasH * scale)
    }

    canvas.width = canvasW
    canvas.height = canvasH

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(img, cropX, cropY, cropW, cropH, 0, 0, canvasW, canvasH)

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `crop_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setCropImageSrc(null)
        onSubmit([file])
      }
    }, 'image/jpeg', 0.83)
  }

  const handleSkipCrop = () => {
    const img = imgRef.current
    if (!img || !imageLoaded) return

    const canvas = document.createElement('canvas')
    const w = imageDims.current.w || img.naturalWidth || img.clientWidth || 800
    const h = imageDims.current.h || img.naturalHeight || img.clientHeight || 600
    
    // Client-side downscaling cap (max 1200px)
    const maxDim = 1200
    let canvasW = w
    let canvasH = h
    if (Math.max(canvasW, canvasH) > maxDim) {
      const scale = maxDim / Math.max(canvasW, canvasH)
      canvasW = Math.round(canvasW * scale)
      canvasH = Math.round(canvasH * scale)
    }
    
    canvas.width = canvasW
    canvas.height = canvasH
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(img, 0, 0, canvasW, canvasH)
    
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `full_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setCropImageSrc(null)
        onSubmit([file])
      }
    }, 'image/jpeg', 0.83)
  }

  return (
    <div className="space-y-6 w-full max-w-3xl mx-auto">
      {/* ── LIVE IN-APP CAMERA SCANNER OVERLAY ── */}
      {showCamera && (
        <div className="fixed inset-0 bg-black z-50 flex flex-col items-center justify-between p-4">
          <div className="w-full flex items-center justify-between text-white py-2">
            <span className="font-medium text-sm">Align invoice within the frame</span>
            <button 
              onClick={stopCamera}
              className="text-white hover:text-gray-300 font-semibold p-2"
            >
              ✕ Close
            </button>
          </div>

          <div className="relative w-full max-w-md flex-1 flex items-center justify-center overflow-hidden rounded-lg bg-neutral-900 border border-neutral-800">
            <video 
              ref={videoRef}
              autoPlay 
              playsInline 
              muted
              className="w-full h-full object-cover"
            />
            {/* Visual Scanning Guide Frame */}
            <div className="absolute border-2 border-dashed border-emerald-400 rounded-lg w-64 h-96 pointer-events-none flex items-center justify-center opacity-70">
              <div className="absolute top-0 left-0 w-full h-0.5 bg-emerald-400 opacity-60 shadow-[0_0_10px_#34d399] animate-pulse"></div>
              <span className="text-emerald-300 text-xs font-semibold bg-black/60 px-3 py-1 rounded-full uppercase tracking-wider">
                Document Frame
              </span>
            </div>
          </div>

          <div className="w-full flex items-center justify-center gap-6 py-6">
            <button 
              onClick={stopCamera} 
              className="px-4 py-2 text-sm font-semibold text-white/80 hover:text-white"
            >
              Cancel
            </button>
            <button 
              onClick={capturePhoto} 
              className="w-16 h-16 rounded-full border-4 border-white bg-white hover:bg-neutral-200 transition-transform active:scale-95 shadow-[0_0_15px_rgba(255,255,255,0.4)]"
              title="Capture Image"
            />
            <div className="w-12 h-12" />
          </div>
        </div>
      )}

      {/* ── CROP INTERFACE OVERLAY ── */}
      {cropImageSrc && (
        <div className="glass p-4 rounded-xl space-y-4 animate-fade-in">
          <div className="flex items-center justify-between border-b border-gray-200 pb-2">
            <div>
              <h3 className="font-bold text-gray-800 text-base">Crop Invoice (Optional)</h3>
              <p className="text-gray-500 text-xs mt-0.5">Drag to select the invoice boundaries to improve accuracy.</p>
            </div>
            <button 
              onClick={() => setCropImageSrc(null)}
              className="text-gray-500 hover:text-gray-700 text-sm font-semibold p-1"
            >
              ✕ Cancel
            </button>
          </div>

          {/* Interactive Image Crop Box Canvas */}
          <div className="flex justify-center bg-gray-100/50 p-2 rounded-lg border border-gray-200">
            <div 
              ref={containerRef}
              onMouseDown={handleDragStart}
              onMouseMove={handleDragMove}
              onMouseUp={handleDragEnd}
              onTouchStart={handleDragStart}
              onTouchMove={handleDragMove}
              onTouchEnd={handleDragEnd}
              className="relative inline-block overflow-hidden cursor-crosshair select-none rounded max-w-full max-h-[60vh]"
              style={{ touchAction: 'none' }}
            >
              <img 
                ref={imgRef}
                src={cropImageSrc} 
                alt="Crop preview" 
                draggable="false"
                onLoad={handleImageLoad}
                className="max-w-full max-h-[60vh] block pointer-events-none"
              />
              
              {/* CSS Box Shadow Mask Overlay */}
              {cropRect.w > 0 && cropRect.h > 0 && (
                <div 
                  className="absolute border-2 border-dashed border-orange-500 pointer-events-none"
                  style={{
                    left: `${cropRect.x}px`,
                    top: `${cropRect.y}px`,
                    width: `${cropRect.w}px`,
                    height: `${cropRect.h}px`,
                    boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.6)', // Dark mask for unselected area
                  }}
                >
                  {/* Handle markers */}
                  <div className="absolute -top-1.5 -left-1.5 w-3 h-3 bg-orange-500 rounded-full border border-white"></div>
                  <div className="absolute -top-1.5 -right-1.5 w-3 h-3 bg-orange-500 rounded-full border border-white"></div>
                  <div className="absolute -bottom-1.5 -left-1.5 w-3 h-3 bg-orange-500 rounded-full border border-white"></div>
                  <div className="absolute -bottom-1.5 -right-1.5 w-3 h-3 bg-orange-500 rounded-full border border-white"></div>
                </div>
              )}
            </div>
            
            {!imageLoaded && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/70 rounded z-10">
                <div className="text-sm font-semibold text-gray-600 animate-pulse">Loading preview...</div>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <button 
              onClick={resetCrop}
              disabled={!imageLoaded || cropRect.w === 0}
              className="btn-ghost text-xs py-1.5 px-3"
            >
              Reset Selection
            </button>
            <div className="flex gap-2">
              <button 
                onClick={handleSkipCrop}
                disabled={!imageLoaded}
                className="btn-ghost text-xs py-2 px-4 border border-gray-300 disabled:opacity-50"
              >
                Skip & Process Full
              </button>
              <button 
                onClick={handleCropConfirm}
                disabled={!imageLoaded || cropRect.w < 20 || cropRect.h < 20}
                className="btn-primary text-xs py-2 px-5 disabled:opacity-50"
              >
                Crop & Process
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── STANDARD LANDING VIEW ── */}
      {!cropImageSrc && (
        <>
          <section className="text-center space-y-2">
            <h1 className="text-3xl md:text-4xl font-semibold text-gray-900 font-sans tracking-tight">Upload Invoice</h1>
            <p className="text-gray-600 text-sm">
              Use the in-app camera scanner or upload image/PDF files to extract invoice details.
            </p>
          </section>

          {error && (
            <div className="p-3 rounded-md border border-red-200 bg-red-50 text-red-700 text-sm">
              {error}
            </div>
          )}

          {cameraError && (
            <div className="p-3 rounded-md border border-yellow-200 bg-yellow-50 text-yellow-800 text-xs">
              {cameraError}
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2">
            {/* Trigger In-App Camera Stream */}
            <div
              onClick={startCamera}
              className={`flex flex-col items-center justify-center gap-3 p-8 rounded-lg border border-gray-300 bg-white cursor-pointer hover:bg-gray-50 transition-colors shadow-sm ${disabled ? 'pointer-events-none opacity-50' : ''}`}
            >
              <div className="p-3 bg-emerald-50 rounded-full text-emerald-600">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-gray-900 font-semibold text-base">Use Camera Scanner</p>
                <p className="text-gray-500 text-xs mt-0.5">Captures & crops for premium VLM parsing</p>
              </div>
              {/* Native mobile camera fallback input (hidden) */}
              <input 
                type="file" 
                ref={cameraInputRef}
                accept="image/*" 
                capture="environment" 
                onChange={onChange} 
                className="hidden" 
                disabled={disabled} 
              />
            </div>

            {/* Trigger File Browse/Drag-and-Drop */}
            <div
              onClick={handleFileClick}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className={`flex flex-col items-center justify-center gap-3 p-8 rounded-lg border-2 border-dashed
                cursor-pointer transition-colors shadow-sm
                ${dragging ? 'border-gray-500 bg-gray-100' : 'border-gray-300 bg-white hover:bg-gray-50'}
                ${disabled ? 'pointer-events-none opacity-50' : ''}`}
            >
              <div className="p-3 bg-blue-50 rounded-full text-blue-600">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-gray-900 font-semibold text-base">Upload Files</p>
                <p className="text-gray-500 text-xs mt-0.5">Select image or PDF files under 10MB</p>
              </div>
              <input 
                type="file" 
                ref={fileInputRef}
                multiple 
                accept={accept} 
                onChange={onChange} 
                className="hidden" 
                disabled={disabled} 
              />
            </div>
          </div>
        </>
      )}
    </div>
  )
}
