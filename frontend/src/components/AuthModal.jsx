import { useState, useRef, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'

export default function AuthModal() {
  const { isAuthModalOpen, closeAuth, login } = useAuth()
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hint, setHint] = useState(null)

  const modalRef = useRef(null)

  useEffect(() => {
    if (isAuthModalOpen) {
      setError(null)
      setHint(null)
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
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })
      const data = await res.json()

      if (!res.ok) {
        if (res.status === 404 && isLogin) {
          setIsLogin(false)
          setHint("Account not found. We've switched you to Sign Up.")
          setLoading(false)
          return
        }
        if (res.status === 409 && !isLogin) {
          setIsLogin(true)
          setHint('Account already exists. Please login with your password.')
          setLoading(false)
          return
        }

        throw new Error(data.detail || 'Authentication failed')
      }

      login(data.token)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/30">
      <div
        ref={modalRef}
        className="w-full max-w-sm bg-white border border-gray-200 rounded-md p-6 relative shadow-lg"
      >
        <button
          onClick={closeAuth}
          className="absolute top-3 right-3 text-gray-500 hover:text-gray-900"
        >
          x
        </button>

        <h2 className="text-2xl font-semibold text-gray-900 mb-5 text-center">
          {isLogin ? 'Welcome Back' : 'Create Account'}
        </h2>

        {error && (
          <div className="mb-4 p-3 rounded-md border border-red-200 bg-red-50 text-red-700 text-sm">
            {error}
          </div>
        )}

        {hint && !error && (
          <div className="mb-4 p-3 rounded-md border border-blue-200 bg-blue-50 text-blue-700 text-sm">
            {hint}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Email Address</label>
            <div>
              <input
                type="email"
                required
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="w-full border border-gray-300 rounded-md py-2.5 px-3 text-sm text-gray-900 focus:outline-none focus:border-gray-500"
                placeholder="you@example.com"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Password</label>
            <div>
              <input
                type="password"
                required
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full border border-gray-300 rounded-md py-2.5 px-3 text-sm text-gray-900 focus:outline-none focus:border-gray-500"
                placeholder="••••••••"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary flex justify-center py-2.5 mt-1"
          >
            {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Sign Up')}
          </button>
        </form>

        <div className="mt-5 text-center text-sm text-gray-600">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button
            onClick={() => { setIsLogin(!isLogin); setError(null); setHint(null); }} 
            className="text-gray-900 font-medium underline"
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </div>
      </div>
    </div>
  )
}
