import ResultTabs from '../ResultTabs'

export default function ResultsScreen({ results, error, onReset }) {
  const successCount = results.filter(r => r.status === 'success').length

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 border border-gray-200 rounded-md bg-white">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Analysis Complete</h2>
          <p className="text-sm text-gray-600">
            Processed {successCount} of {results.length} document{results.length !== 1 ? 's' : ''}.
          </p>
        </div>

        <button onClick={onReset} className="btn-primary whitespace-nowrap">
          Scan Another
        </button>
      </div>

      {error && (
        <div className="p-3 rounded-md border border-red-200 bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}

      <div className="glass p-4">
        <ResultTabs results={results} />
      </div>
    </div>
  )
}
