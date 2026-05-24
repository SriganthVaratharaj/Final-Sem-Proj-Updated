import HomePage from './pages/HomePage'
import AuthModal from './components/AuthModal'
import { AuthProvider, useAuth } from './context/AuthContext'

function NavBar() {
  const { user, openAuth, logout } = useAuth()

  return (
    <nav className="border-b border-gray-200 bg-white sticky top-0 z-40">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="font-semibold text-lg text-gray-900">Invoice Extractor</div>
        <div className="flex items-center gap-4">
          {user ? (
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-600 font-medium">{user.email}</span>
              <button onClick={logout} className="btn-ghost text-xs py-1 px-3 border border-gray-300 rounded hover:bg-gray-100 transition-colors">
                Sign Out
              </button>
            </div>
          ) : (
            <button onClick={openAuth} className="btn-primary text-xs py-1.5 px-4 rounded font-medium">
              Sign In
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
      <div className="min-h-screen bg-white text-gray-900 font-sans">
        <NavBar />
        <HomePage />

        <footer className="mt-14 py-6 border-t border-gray-200 text-center text-xs text-gray-500">
          Combined invoice extractor using OCR, layout analysis, and VLM.
        </footer>

        <AuthModal />
      </div>
    </AuthProvider>
  )
}
