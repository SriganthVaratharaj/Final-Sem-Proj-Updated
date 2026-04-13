// frontend/src/components/screens/ResultsScreen.jsx
import ResultTabs from '../ResultTabs'
import { Plus, CheckCircle } from 'lucide-react'

export default function ResultsScreen({ results, error, onReset }) {
  const successCount = results.filter(r => r.status === 'success').length

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fade-in relative z-10">
      
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-5 glass rounded-2xl border border-emerald-900/50 bg-emerald-900/10">
        <div className="flex items-center gap-3">
          <div className="bg-emerald-500/20 p-2 rounded-full">
            <CheckCircle size={24} className="text-emerald-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-emerald-300">Analysis Complete</h2>
            <p className="text-sm text-emerald-400/80">
              Successfully processed {successCount} of {results.length} document{results.length !== 1 ? 's' : ''}.
            </p>
          </div>
        </div>
        
        <button 
          onClick={onReset}
          className="btn-primary flex items-center gap-2 whitespace-nowrap bg-brand-600 hover:bg-brand-500 text-white"
        >
          <Plus size={18} />
          Scan Another
        </button>
      </div>

      {/* Main Results Container */}
      <ResultTabs results={results} />
    </div>
  )
}
