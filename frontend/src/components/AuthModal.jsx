// frontend/src/components/AuthModal.jsx
import { useState, useRef, useEffect } from 'react'
import { X, Mail, Lock, Loader2, AlertCircle } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function AuthModal() {
  const { isAuthModalOpen, closeAuth, login } = useAuth()
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const modalRef = useRef(null)

  // Quick reset on open
  useEffect(() => {
    if (isAuthModalOpen) {
      setError(null)
      setPassword('')
    }
  }, [isAuthModalOpen])

  if (!isAuthModalOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register'
    
    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })
      const data = await res.json()

      if (!res.ok) {
        // Handle specific logic requested by user:
        if (res.status === 404 && isLogin) {
            setIsLogin(false)
            setError("Account does not exist. Please sign up.")
            setLoading(false)
            return
        }
        if (res.status === 409 && !isLogin) {
            setIsLogin(true)
            setError("Account already exists. Please login instead.")
            setLoading(false)
            return
        }
        
        throw new Error(data.detail || "Authentication failed")
      }

      // Success
      login(data.token)

    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div 
        ref={modalRef} 
        className="w-full max-w-sm glass bg-gray-900 border border-gray-700/50 rounded-3xl p-6 relative"
      >
        <button 
          onClick={closeAuth}
          className="absolute top-4 right-4 text-gray-500 hover:text-white transition"
        >
          <X size={20} />
        </button>

        <h2 className="text-2xl font-bold text-white mb-6 text-center">
          {isLogin ? 'Welcome Back' : 'Create Account'}
        </h2>

        {error && (
          <div className="mb-4 p-3 rounded-xl border border-red-900/50 bg-red-900/20 text-red-400 text-sm flex items-start gap-2">
            <AlertCircle size={16} className="shrink-0 mt-0.5" />
            <span className="flex-1 leading-snug">{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1 ml-1">Email Address</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input 
                type="email" 
                required
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-700 rounded-xl py-2.5 pl-9 pr-4 text-sm text-white focus:outline-none focus:border-brand-500 transition-colors"
                placeholder="you@example.com"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1 ml-1">Password</label>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input 
                type="password" 
                required
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-700 rounded-xl py-2.5 pl-9 pr-4 text-sm text-white focus:outline-none focus:border-brand-500 transition-colors"
                placeholder="••••••••"
              />
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full btn-primary bg-brand-600 hover:bg-brand-500 text-white flex justify-center py-3 mt-2"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : (isLogin ? 'Sign In' : 'Sign Up')}
          </button>
        </form>

        <div className="mt-6 text-center text-sm text-gray-400">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            onClick={() => { setIsLogin(!isLogin); setError(null); }} 
            className="text-brand-400 hover:text-brand-300 font-semibold transition"
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </div>
      </div>
    </div>
  )
}
