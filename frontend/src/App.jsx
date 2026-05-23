import HomePage from './pages/HomePage'
import AuthModal from './components/AuthModal'
import { AuthProvider, useAuth } from './context/AuthContext'

function NavBar() {
  const { user, openAuth, logout } = useAuth()

  return (
    <nav className="border-b border-gray-200 bg-white sticky top-0 z-40">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="font-semibold text-lg text-gray-900">Invoice Extractor</div>

        {/* Sign In option is hidden for now */}
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
