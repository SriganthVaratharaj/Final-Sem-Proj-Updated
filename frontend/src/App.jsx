// frontend/src/App.jsx
import { Sparkles, FileText, LogOut, UserCircle, Scan } from 'lucide-react'
import HomePage from './pages/HomePage'
import AuthModal from './components/AuthModal'
import { AuthProvider, useAuth } from './context/AuthContext'

function NavBar() {
  const { user, openAuth, logout } = useAuth()
  
  return (
    <nav className="border-b border-gray-800/50 bg-gray-950/40 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-brand-600 rounded-xl">
            <Scan size={20} className="text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight text-white flex items-center gap-2">
            Invoice AI <span className="px-2 py-0.5 rounded-full bg-brand-900/50 text-brand-400 text-xs border border-brand-700/50">v2.0</span>
          </span>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-sm font-medium text-gray-400">
            <span className="hover:text-brand-300 transition cursor-help" title="Module 1">PaddleOCR</span>
            <span className="opacity-50 text-xs">•</span>
            <span className="hover:text-emerald-300 transition cursor-help" title="Module 2">LayoutLMv3</span>
            <span className="opacity-50 text-xs">•</span>
            <span className="hover:text-purple-300 transition cursor-help" title="Module 3">LLaVA</span>
          </div>

          <div className="h-6 w-px bg-gray-800 hidden md:block"></div>

          {user ? (
            <div className="flex items-center gap-3">
              <span className="hidden sm:inline-block text-xs font-medium text-emerald-400 bg-emerald-900/20 py-1 px-3 rounded-full border border-emerald-800/50">
                {user.email}
              </span>
              <button onClick={logout} className="text-gray-400 hover:text-white transition" title="Logout">
                <LogOut size={18} />
              </button>
            </div>
          ) : (
            <button 
              onClick={openAuth}
              className="flex items-center gap-2 text-sm font-semibold text-brand-300 hover:text-white bg-brand-900/30 hover:bg-brand-600/50 py-1.5 px-4 border border-brand-700/50 rounded-full transition-all"
            >
              <UserCircle size={16} /> Sign In
            </button>
          )}
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-[#050505] text-gray-200 font-sans selection:bg-brand-500/30">
        <NavBar />
        <HomePage />
        
        <footer className="mt-20 py-8 border-t border-gray-900 text-center text-xs text-gray-500">
          Combined Invoice Extractor • Uses PaddleOCR, HuggingFace Inference API, and MongoDB
        </footer>

        <AuthModal />
      </div>
    </AuthProvider>
  )
}
