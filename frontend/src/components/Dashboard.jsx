import { useState, useEffect } from 'react'
import { fetchHistory, searchHistory, deleteHistoryItem } from '../services/api'
import { useAuth } from '../context/AuthContext'
import { ResultCard } from './ResultTabs'

// Helper to safely extract amount from invoice fields
const parseAmount = (fields) => {
  if (!fields) return 0
  const keys = ['total_amount', 'grand_total', 'total', 'amount_due', 'amount']
  for (const k of keys) {
    if (fields[k]) {
      const str = String(fields[k])
      const match = str.replace(/[,₹$]/g, '').match(/\d+(?:\.\d+)?/)
      if (match) return parseFloat(match[0])
    }
  }
  for (const [k, v] of Object.entries(fields)) {
    if (k.toLowerCase().includes('total') || k.toLowerCase().includes('amount')) {
      const str = String(v)
      const match = str.replace(/[,₹$]/g, '').match(/\d+(?:\.\d+)?/)
      if (match) return parseFloat(match[0])
    }
  }
  return 0
}

// Helper to extract vendor name
const getVendor = (fields) => {
  if (!fields) return 'Unknown'
  const keys = ['vendor_name', 'company_name', 'merchant', 'vendor', 'name']
  for (const k of keys) {
    if (fields[k] && fields[k] !== 'Not detected' && fields[k] !== 'null') {
      return fields[k]
    }
  }
  return 'Other'
}

export default function Dashboard() {
  const { token } = useAuth()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [error, setError] = useState(null)

  // Filters State
  const [filterVendor, setFilterVendor] = useState('all')
  const [filterDocType, setFilterDocType] = useState('all')
  const [filterLanguage, setFilterLanguage] = useState('all')
  const [filterMinAmount, setFilterMinAmount] = useState('')
  const [filterMaxAmount, setFilterMaxAmount] = useState('')

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchHistory(100, token)
      setHistory(data.results || [])
    } catch (err) {
      console.error('Failed to load history:', err)
      setError('Could not connect to database to fetch analytics.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [token])

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!searchQuery.trim()) {
      setSearchResults(null)
      return
    }
    setSearchLoading(true)
    try {
      const data = await searchHistory(searchQuery, token)
      setSearchResults(data.results || [])
    } catch (err) {
      console.error('Failed to search:', err)
    } finally {
      setSearchLoading(false)
    }
  }

  const handleClearSearch = () => {
    setSearchQuery('')
    setSearchResults(null)
  }

  const handleResetFilters = () => {
    setFilterVendor('all')
    setFilterDocType('all')
    setFilterLanguage('all')
    setFilterMinAmount('')
    setFilterMaxAmount('')
  }

  // --- FILTER & ANALYTICS CALCULATIONS ---
  const successDocs = history.filter((h) => h.status === 'success')

  // Extract unique filter values from all successful docs
  const allVendors = Array.from(new Set(successDocs.map(doc => {
    const fields = doc.vlm?.fields || doc.vlm_fields || {}
    return getVendor(fields)
  }))).filter(v => v && v !== 'Other' && v !== 'Unknown').sort()
  
  if (successDocs.some(doc => {
    const v = getVendor(doc.vlm?.fields || doc.vlm_fields || {})
    return v === 'Other' || v === 'Unknown'
  })) {
    allVendors.push('Other')
  }

  const allDocTypes = Array.from(new Set(successDocs.map(doc => doc.document_type || 'invoice'))).filter(Boolean).sort()
  
  const allLanguages = Array.from(new Set(successDocs.map(doc => {
    const lang = doc.metadata?.dominant_language || doc.metadata?.detected_language || 'unknown'
    return lang.toLowerCase()
  }))).filter(Boolean).sort()

  // Apply filters
  const filteredDocs = successDocs.filter((doc) => {
    const fields = doc.vlm?.fields || doc.vlm_fields || {}
    const vendor = getVendor(fields)
    const amount = parseAmount(fields)
    const docType = doc.document_type || 'invoice'
    const lang = (doc.metadata?.dominant_language || doc.metadata?.detected_language || 'unknown').toLowerCase()

    if (filterVendor !== 'all' && vendor !== filterVendor) return false
    if (filterDocType !== 'all' && docType !== filterDocType) return false
    if (filterLanguage !== 'all' && lang !== filterLanguage) return false
    if (filterMinAmount !== '' && amount < parseFloat(filterMinAmount)) return false
    if (filterMaxAmount !== '' && amount > parseFloat(filterMaxAmount)) return false
    return true
  })

  const totalInvoices = filteredDocs.length
  
  // Calculate total expense for filtered docs
  const totalExpense = filteredDocs.reduce((acc, doc) => {
    const fields = doc.vlm?.fields || doc.vlm_fields || {}
    return acc + parseAmount(fields)
  }, 0)

  // Group by Monthly spending (filtered docs)
  const monthlyMap = {}
  filteredDocs.forEach((doc) => {
    if (!doc.created_at) return
    const date = new Date(doc.created_at)
    const monthKey = date.toLocaleString('default', { month: 'short', year: '2-digit' })
    const fields = doc.vlm?.fields || doc.vlm_fields || {}
    const amt = parseAmount(fields)
    monthlyMap[monthKey] = (monthlyMap[monthKey] || 0) + amt
  })

  const monthlySpending = Object.entries(monthlyMap)
    .map(([month, amount]) => ({ month, amount }))
    .slice(-6) // last 6 months

  const maxMonthlyAmount = Math.max(...monthlySpending.map((m) => m.amount), 100)

  // Calculate top vendors using all filters EXCEPT the vendor filter
  // This allows the user to see how the selected vendor compares to other top vendors
  const docsForVendorChart = successDocs.filter((doc) => {
    const fields = doc.vlm?.fields || doc.vlm_fields || {}
    const amount = parseAmount(fields)
    const docType = doc.document_type || 'invoice'
    const lang = (doc.metadata?.dominant_language || doc.metadata?.detected_language || 'unknown').toLowerCase()

    if (filterDocType !== 'all' && docType !== filterDocType) return false
    if (filterLanguage !== 'all' && lang !== filterLanguage) return false
    if (filterMinAmount !== '' && amount < parseFloat(filterMinAmount)) return false
    if (filterMaxAmount !== '' && amount > parseFloat(filterMaxAmount)) return false
    return true
  })

  const vendorChartMap = {}
  docsForVendorChart.forEach((doc) => {
    const fields = doc.vlm?.fields || doc.vlm_fields || {}
    const vendor = getVendor(fields)
    const amt = parseAmount(fields)
    vendorChartMap[vendor] = (vendorChartMap[vendor] || 0) + amt
  })

  const topVendors = Object.entries(vendorChartMap)
    .map(([name, amount]) => ({ name, amount }))
    .sort((a, b) => b.amount - a.amount)
    .slice(0, 5)

  const maxVendorAmount = Math.max(...topVendors.map((v) => v.amount), 100)

  return (
    <div className="space-y-6 animate-fade-in pb-10">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass p-5 flex flex-col justify-between">
          <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Total Expenses Logged</span>
          <h2 className="text-3xl font-extrabold text-gray-900 mt-2">
            ₹{totalExpense.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
          </h2>
          <span className="text-xs text-gray-500 mt-2">Aggregated from filtered extractions</span>
        </div>

        <div className="glass p-5 flex flex-col justify-between">
          <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Invoices Processed</span>
          <h2 className="text-3xl font-extrabold text-gray-900 mt-2">{totalInvoices}</h2>
          <span className="text-xs text-green-600 mt-2 font-medium">✓ Mapped securely in database</span>
        </div>

        <div className="glass p-5 flex flex-col justify-between">
          <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Top Expense Vendor</span>
          <h2 className="text-xl font-bold text-gray-900 mt-2 truncate">
            {topVendors[0] ? `${topVendors[0].name} (₹${topVendors[0].amount.toLocaleString('en-IN')})` : 'None Detected'}
          </h2>
          <span className="text-xs text-gray-500 mt-2 font-medium">Click vendor below to quick-filter</span>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-10">
          <div className="text-sm font-semibold text-gray-600 animate-pulse">Loading financial intelligence...</div>
        </div>
      ) : error ? (
        <div className="p-4 rounded-md border border-red-200 bg-red-50 text-red-700 text-sm">{error}</div>
      ) : (
        <>
          {/* Filter Controls Panel */}
          <div className="glass p-5 space-y-4">
            <div className="flex justify-between items-center border-b border-gray-100 pb-2">
              <div>
                <h3 className="text-sm font-bold text-gray-800">Financial Filter Panel</h3>
                <p className="text-[11px] text-gray-500 mt-0.5">Filter analytics by merchant, type, language, or values</p>
              </div>
              {(filterVendor !== 'all' || filterDocType !== 'all' || filterLanguage !== 'all' || filterMinAmount !== '' || filterMaxAmount !== '') && (
                <button
                  onClick={handleResetFilters}
                  className="text-xs font-semibold text-red-600 hover:text-red-800 transition-colors"
                >
                  Reset Filters
                </button>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {/* Vendor select */}
              <div>
                <label className="block text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1">Vendor</label>
                <select
                  value={filterVendor}
                  onChange={(e) => setFilterVendor(e.target.value)}
                  className="w-full border border-gray-300 rounded-md py-1.5 px-2 text-xs text-gray-900 bg-white focus:outline-none focus:border-gray-500"
                >
                  <option value="all">All Vendors</option>
                  {allVendors.map((v, i) => (
                    <option key={i} value={v}>{v}</option>
                  ))}
                </select>
              </div>

              {/* Doc Type select */}
              <div>
                <label className="block text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1">Doc Type</label>
                <select
                  value={filterDocType}
                  onChange={(e) => setFilterDocType(e.target.value)}
                  className="w-full border border-gray-300 rounded-md py-1.5 px-2 text-xs text-gray-900 bg-white focus:outline-none focus:border-gray-500"
                >
                  <option value="all">All Types</option>
                  {allDocTypes.map((t, i) => (
                    <option key={i} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              {/* Language select */}
              <div>
                <label className="block text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1">Language</label>
                <select
                  value={filterLanguage}
                  onChange={(e) => setFilterLanguage(e.target.value)}
                  className="w-full border border-gray-300 rounded-md py-1.5 px-2 text-xs text-gray-900 bg-white focus:outline-none focus:border-gray-500"
                >
                  <option value="all">All Languages</option>
                  {allLanguages.map((l, i) => (
                    <option key={i} value={l}>{l.charAt(0).toUpperCase() + l.slice(1)}</option>
                  ))}
                </select>
              </div>

              {/* Min Amount input */}
              <div>
                <label className="block text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1">Min Amount</label>
                <input
                  type="number"
                  value={filterMinAmount}
                  onChange={(e) => setFilterMinAmount(e.target.value)}
                  placeholder="e.g. 0"
                  className="w-full border border-gray-300 rounded-md py-1.5 px-2 text-xs text-gray-900 focus:outline-none focus:border-gray-500"
                />
              </div>

              {/* Max Amount input */}
              <div>
                <label className="block text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1">Max Amount</label>
                <input
                  type="number"
                  value={filterMaxAmount}
                  onChange={(e) => setFilterMaxAmount(e.target.value)}
                  placeholder="e.g. 10000"
                  className="w-full border border-gray-300 rounded-md py-1.5 px-2 text-xs text-gray-900 focus:outline-none focus:border-gray-500"
                />
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500 font-medium pt-1">
              <span>Filter matches: <strong className="text-gray-800">{totalInvoices}</strong> / {successDocs.length} invoices</span>
              {filterVendor !== 'all' && (
                <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded text-[10px] font-semibold">
                  Filtered to: {filterVendor}
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Monthly Spending Trend Custom SVG Chart */}
            <div className="glass p-5 space-y-4">
              <div>
                <h3 className="text-sm font-bold text-gray-800">Monthly Spending Trend</h3>
                <p className="text-[11px] text-gray-500 mt-0.5">Aggregated expense history over time</p>
              </div>
              {monthlySpending.length === 0 ? (
                <div className="h-48 flex items-center justify-center text-xs text-gray-400 italic">No spending data logged for active filters.</div>
              ) : (
                <div className="relative pt-4">
                  <div className="flex items-end justify-between gap-2 h-44 border-b border-gray-200 pb-2">
                    {monthlySpending.map((item, idx) => {
                      const heightPercent = (item.amount / maxMonthlyAmount) * 80 + 10 // scale to max 90% height
                      return (
                        <div key={idx} className="flex-1 flex flex-col items-center group relative cursor-pointer">
                          {/* Tooltip */}
                          <div className="absolute bottom-full mb-1 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 text-white text-[10px] px-2 py-0.5 rounded pointer-events-none whitespace-nowrap shadow z-10">
                            ₹{item.amount.toLocaleString()}
                          </div>
                          {/* Bar */}
                          <div
                            style={{ height: `${heightPercent}%` }}
                            className="w-full bg-neutral-900 hover:bg-neutral-800 rounded-t-md transition-all duration-300 shadow-sm"
                          ></div>
                          <span className="text-[10px] text-gray-500 font-semibold mt-2">{item.month}</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Top Vendors Horizontal SVG Progress Bars */}
            <div className="glass p-5 space-y-4">
              <div>
                <h3 className="text-sm font-bold text-gray-800">Top Vendors spending share</h3>
                <p className="text-[11px] text-gray-500 mt-0.5">Primary merchants (Click row to quick-filter)</p>
              </div>
              {topVendors.length === 0 ? (
                <div className="h-48 flex items-center justify-center text-xs text-gray-400 italic">No vendor transactions detected.</div>
              ) : (
                <div className="space-y-2 pt-1">
                  {topVendors.map((vendor, idx) => {
                    const widthPercent = (vendor.amount / maxVendorAmount) * 100
                    const isSelected = filterVendor === vendor.name
                    return (
                      <div 
                        key={idx} 
                        onClick={() => setFilterVendor(filterVendor === vendor.name ? 'all' : vendor.name)}
                        className={`space-y-1 p-2 rounded-md cursor-pointer transition-all duration-200 border ${
                          isSelected 
                            ? 'bg-indigo-50/75 border-indigo-200 shadow-sm' 
                            : 'hover:bg-gray-50/75 border-transparent'
                        }`}
                      >
                        <div className="flex justify-between text-xs font-semibold text-gray-800">
                          <span className="truncate max-w-[70%] flex items-center gap-1.5">
                            {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-ping"></span>}
                            {vendor.name}
                          </span>
                          <span>₹{vendor.amount.toLocaleString()}</span>
                        </div>
                        <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${widthPercent}%` }}
                            className={`h-full rounded-full transition-all duration-500 ${
                              isSelected ? 'bg-indigo-700' : 'bg-indigo-600'
                            }`}
                          ></div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* ── SMART SEARCH MODULE ── */}
      <div className="glass p-5 space-y-4">
        <div>
          <h3 className="text-sm font-bold text-gray-800">Smart Document Search</h3>
          <p className="text-[11px] text-gray-500 mt-0.5 font-medium">
            Search invoices instantly by company name, total, currency, language, or raw text content.
          </p>
        </div>

        {/* Search Bar Form */}
        <form onSubmit={handleSearch} className="flex gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Type invoice keyword (e.g. Electricity, ₹1700, English, etc.)..."
              className="w-full border border-gray-300 rounded-md py-2 px-3 pl-9 text-sm text-gray-900 focus:outline-none focus:border-gray-500"
            />
            <svg
              className="absolute left-3 top-2.5 w-4.5 h-4.5 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <button type="submit" disabled={searchLoading} className="btn-primary text-xs py-2 px-5 shrink-0">
            {searchLoading ? 'Searching...' : 'Search'}
          </button>
          {searchResults !== null && (
            <button
              type="button"
              onClick={handleClearSearch}
              className="btn-ghost text-xs py-2 px-4 shrink-0 border border-gray-300 rounded hover:bg-gray-100"
            >
              Clear
            </button>
          )}
        </form>

        {/* Search Results rendering */}
        {searchResults !== null && (
          <div className="space-y-4 pt-2 border-t border-gray-100">
            <div className="text-xs text-gray-500 font-semibold">Found {searchResults.length} matching documents</div>
            {searchResults.length === 0 ? (
              <div className="text-center py-6 text-sm text-gray-500 italic">No matching results. Try another term!</div>
            ) : (
              <div className="space-y-4">
                {searchResults.map((r, i) => (
                  <ResultCard key={i} result={r} defaultOpen={false} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
