import { useState } from 'react'
import { getFileUrl } from '../services/api'

const SOURCE_META = {
  kaggle_remote_vlm: { label: 'Kaggle Qwen2.5-VL-32B (Remote Tunnel)', tier: '30GB VRAM' },
  llava_local_with_ocr_hint: { label: 'Local CUDA Qwen-VL (with OCR Hint)', tier: 'GPU' },
  llava_local_only: { label: 'Local CUDA Qwen-VL (Visual Only)', tier: 'GPU' },
  image_only_fallback: { label: 'Local Fallback (Visual Only)', tier: 'GPU' },
  llava: { label: 'LLaVA 1.6 Mistral-7B', tier: 'Tier 1' },
  blip2: { label: 'BLIP-2 Flan-T5-XL', tier: 'Tier 2' },
  local: { label: 'BLIP-base (local)', tier: 'Tier 3' },
  unavailable: { label: 'Model Unavailable', tier: '' },
}

function handleDownloadTxt(result, translatedText = null, targetLang = null) {
  const rawFields = result.vlm_fields || {}
  
  let text = `DOCUMENT EXTRACTION REPORT\n`
  text += `==========================\n\n`
  text += `File Name      : ${result.image_name || 'report'}\n`
  text += `Classification : ${result.metadata?.classification || 'Document'}\n`
  if (rawFields.metadata?.detected_language) {
    text += `Language       : ${rawFields.metadata.detected_language}\n`
  }
  text += `\n`

  if (rawFields.full_extraction) {
    text += `=========================================\n`
    text += `NATIVE SCRIPT OUTPUT\n`
    text += `=========================================\n`
    text += rawFields.full_extraction + `\n\n`
  }

  if (rawFields.english_extraction) {
    text += `=========================================\n`
    text += `ENGLISH TRANSLATION OUTPUT\n`
    text += `=========================================\n`
    text += rawFields.english_extraction + `\n\n`
  }

  if (translatedText && targetLang) {
    text += `=========================================\n`
    text += `CUSTOM TRANSLATION (${targetLang.toUpperCase()})\n`
    text += `=========================================\n`
    text += translatedText + `\n\n`
  }

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${result.image_name ? result.image_name.replace(/\.[^/.]+$/, "") : 'report'}_analysis_report.txt`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export function ResultCard({ result, defaultOpen }) {
  const [expanded, setExpanded] = useState(defaultOpen)
  const success = result.status === 'success'

  const rawFields = result.vlm_fields || {}
  const source = result.vlm_source || 'unavailable'
  const meta = SOURCE_META[source] || SOURCE_META.unavailable

  // Custom Translation State
  const [targetLang, setTargetLang] = useState('Tamil')
  const [translating, setTranslating] = useState(false)
  const [translatedText, setTranslatedText] = useState(null)
  const [translateError, setTranslateError] = useState(null)

  const handleTranslate = async () => {
    setTranslating(true)
    setTranslateError(null)
    try {
      const endpoint = getFileUrl('/api/translate')
      const textToTranslate = rawFields.english_extraction || rawFields.full_extraction || result.text_report_preview || ''
      if (!textToTranslate.trim()) {
        throw new Error('No text found in extraction output to translate.')
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToTranslate, target_lang: targetLang })
      })

      if (!res.ok) {
        throw new Error(`Translation failed with status ${res.status}`)
      }

      const data = await res.json()
      setTranslatedText(data.translated_text)
    } catch (err) {
      console.error(err)
      setTranslateError(err.message || 'Translation failed')
    } finally {
      setTranslating(false)
    }
  }

  const handleDownloadTranslation = () => {
    if (!translatedText) return
    let text = `TRANSLATED EXTRACTION REPORT (${targetLang.toUpperCase()})\n`
    text += `==============================================\n\n`
    text += `File Name      : ${result.image_name || 'report'}\n`
    text += `Target Language: ${targetLang}\n\n`
    text += `==============================================\n`
    text += translatedText + `\n`

    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${result.image_name ? result.image_name.replace(/\.[^/.]+$/, "") : 'report'}_translated_${targetLang.toLowerCase()}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="border border-gray-200 rounded-md overflow-hidden bg-white shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 p-4 text-left hover:bg-gray-50 transition-colors"
      >
        <span
          className={`mt-1 h-2.5 w-2.5 rounded-full shrink-0 ${success ? 'bg-green-500' : 'bg-red-500'}`}
        ></span>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-gray-900 truncate">{result.image_name}</div>
          {result.error && <div className="text-xs text-red-600 mt-0.5">{result.error}</div>}
          {success && (
            <div className="flex flex-wrap gap-1.5 mt-1">
              <span className="tag capitalize">{result.metadata?.classification || 'Document'}</span>
              {result.metadata?.detected_language && (
                <span className="tag">{result.metadata.detected_language}</span>
              )}
              {result.vlm_source && result.vlm_source !== 'unavailable' && (
                <span className="tag">Kaggle Node</span>
              )}
            </div>
          )}
        </div>
        <span className="text-gray-500 text-xs shrink-0">{expanded ? 'Hide' : 'Show'}</span>
      </button>

      {expanded && success && (
        <div className="border-t border-gray-200 p-4 space-y-6">
          {/* VLM Inference Engine Info */}
          <div className="glass-light p-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">VLM Inference Engine</div>
              <p className="text-sm text-gray-900 font-semibold mt-0.5">{meta.label}</p>
            </div>
            <div className="flex gap-2">
              {rawFields.metadata?.detected_language && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200">
                  Lang: {rawFields.metadata.detected_language}
                </span>
              )}
              {meta.tier && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-50 text-indigo-800 border border-indigo-200">
                  {meta.tier}
                </span>
              )}
            </div>
          </div>

          {/* Download TXT Report Section */}
          {(rawFields.full_extraction || rawFields.english_extraction || result.text_report_preview) && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 bg-blue-50/50 border border-blue-100 rounded-lg">
              <div>
                <div className="font-semibold text-blue-900 text-sm">Download Report</div>
                <div className="text-xs text-blue-700 mt-0.5">Save the structured text report of the document.</div>
              </div>
              <button
                onClick={() => handleDownloadTxt(result, translatedText, targetLang)}
                className="w-full sm:w-auto flex items-center justify-center gap-2 bg-gray-900 text-white font-medium px-4 py-2 rounded-md hover:bg-gray-800 transition-colors shadow-sm text-sm"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download TXT
              </button>
            </div>
          )}

          {/* Spatial OCR Reconstructions */}
          {rawFields && (rawFields.full_extraction || rawFields.english_extraction) && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="h-4 w-1 bg-gray-900 rounded-full"></div>
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider">Spatial OCR Reconstructions</h3>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {rawFields.full_extraction && (
                  <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <h4 className="text-xs font-bold text-gray-500 mb-2 uppercase tracking-wide">Native Script Output</h4>
                    <pre className="text-[11px] text-gray-800 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto max-h-60 overflow-y-auto">
                      {rawFields.full_extraction}
                    </pre>
                  </div>
                )}
                {rawFields.english_extraction && (
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <h4 className="text-xs font-bold text-blue-600 mb-2 uppercase tracking-wide">English Translation Output</h4>
                    <pre className="text-[11px] text-blue-900 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto max-h-60 overflow-y-auto">
                      {rawFields.english_extraction}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Custom Translation Section */}
          {false && (
            <div className="border-t border-gray-100 pt-5 space-y-3">
              <div className="flex items-center gap-2">
                <div className="h-4 w-1 bg-indigo-500 rounded-full"></div>
                <h3 className="text-xs font-bold text-gray-700 uppercase tracking-wider">Custom Document Translator</h3>
              </div>
              <p className="text-[11px] text-gray-500">Translate the extracted document fields and values into any language using the visual reasoning engine.</p>
              
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-gray-600">Target Language:</span>
                  <select
                    value={targetLang}
                    onChange={(e) => setTargetLang(e.target.value)}
                    className="border border-gray-300 rounded px-2.5 py-1 text-xs bg-white text-gray-800 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="Tamil">Tamil (தமிழ்)</option>
                    <option value="Hindi">Hindi (हिन्दी)</option>
                    <option value="Telugu">Telugu (తెలుగు)</option>
                    <option value="Kannada">Kannada (ಕನ್ನಡ)</option>
                    <option value="Malayalam">Malayalam (മലയാളം)</option>
                    <option value="Bengali">Bengali (বাংলা)</option>
                    <option value="Gujarati">Gujarati (ગુજરાતી)</option>
                    <option value="Marathi">Marathi (मराठी)</option>
                    <option value="Punjabi">Punjabi (ਪੰਜਾਬী)</option>
                    <option value="Spanish">Spanish (Español)</option>
                    <option value="French">French (Français)</option>
                    <option value="German">German (Deutsch)</option>
                    <option value="Japanese">Japanese (日本語)</option>
                    <option value="Chinese">Chinese (中文)</option>
                  </select>
                </div>

                <button
                  onClick={handleTranslate}
                  disabled={translating}
                  className="btn-primary text-xs py-1.5 px-4 rounded font-semibold shrink-0"
                >
                  {translating ? 'Translating...' : 'Translate'}
                </button>
              </div>

              {translating && (
                <div className="text-xs text-indigo-600 font-semibold animate-pulse pt-1">
                  Translating report structure via remote worker node...
                </div>
              )}

              {translateError && (
                <div className="text-xs text-red-600 border border-red-100 bg-red-50 p-2 rounded">
                  Error: {translateError}
                </div>
              )}

              {translatedText && (
                <div className="space-y-2 pt-1">
                  <div className="flex items-center justify-between">
                    <h4 className="text-[11px] font-bold text-gray-500 uppercase tracking-wide">Translated Output ({targetLang})</h4>
                    <button
                      onClick={handleDownloadTranslation}
                      className="text-[10px] text-indigo-600 hover:text-indigo-800 font-bold underline"
                    >
                      Download Translated TXT
                    </button>
                  </div>
                  <pre className="bg-indigo-50/50 p-4 rounded-lg border border-indigo-100 text-[11px] text-indigo-900 font-mono whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                    {translatedText}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ResultTabs({ results }) {
  if (!results || results.length === 0) return null

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-gray-900">Results ({results.length})</h2>
      {results.map((r, i) => (
        <ResultCard key={i} result={r} defaultOpen={i === 0} />
      ))}
    </div>
  )
}
