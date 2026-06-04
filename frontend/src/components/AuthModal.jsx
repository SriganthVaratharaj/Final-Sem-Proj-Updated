import { useState, useRef, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { getFileUrl } from '../services/api'

export default function AuthModal() {
  const { isAuthModalOpen, closeAuth, login } = useAuth()
  
  // view can be 'login', 'register', 'forgot_password', 'verify_otp'
  const [view, setView] = useState('login')
  
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [otp, setOtp] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hint, setHint] = useState(null)
  const [devOtp, setDevOtp] = useState(null)

  const modalRef = useRef(null)

  useEffect(() => {
    if (isAuthModalOpen) {
      setError(null)
      setHint(null)
      setPassword('')
      setOtp('')
      setDevOtp(null)
      setView('login')
    }
  }, [isAuthModalOpen])

  if (!isAuthModalOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    let path = ''
    let body = {}

    if (view === 'login') {
      path = '/api/auth/login'
      body = { email, password }
    } else if (view === 'register') {
      path = '/api/auth/register'
      body = { email, password }
    } else if (view === 'forgot_password') {
      path = '/api/auth/forgot-password'
      body = { email }
    } else if (view === 'verify_otp') {
      path = '/api/auth/reset-password'
      body = { email, otp, new_password: password }
    }

    const endpoint = getFileUrl(path)
    
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      let data = {}
      const contentType = res.headers.get("content-type")
      if (contentType && contentType.includes("application/json")) {
        data = await res.json()
      } else {
        const text = await res.text()
        throw new Error(text || `Request failed with status ${res.status}`)
      }

      if (!res.ok) {
        if (res.status === 404 && view === 'login') {
          setView('register')
          setHint("Account not found. We've switched you to Sign Up.")
          setLoading(false)
          return
        }
        if (res.status === 409 && view === 'register') {
          setView('login')
          setHint('Account already exists. Please login with your password.')
          setLoading(false)
          return
        }

        throw new Error(data.detail || 'Authentication failed')
      }

      if (view === 'login' || view === 'register') {
        login(data.token)
      } else if (view === 'forgot_password') {
        setView('verify_otp')
        setHint('An OTP has been sent to your email. Please enter it below to reset your password.')
        if (data.dev_otp) {
          setDevOtp(data.dev_otp) // For local testing without real email
        }
      } else if (view === 'verify_otp') {
        setView('login')
        setHint('Password reset successfully. Please sign in with your new password.')
        setPassword('')
        setOtp('')
      }
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
          {view === 'login' && 'Welcome Back'}
          {view === 'register' && 'Create Account'}
          {view === 'forgot_password' && 'Reset Password'}
          {view === 'verify_otp' && 'Verify OTP'}
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

        {devOtp && view === 'verify_otp' && (
          <div className="mb-4 p-3 rounded-md border border-yellow-200 bg-yellow-50 text-yellow-800 text-sm font-medium">
            [Testing Mode] Your OTP is: {devOtp}
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
                disabled={view === 'verify_otp'}
                onChange={e => setEmail(e.target.value)}
                className="w-full border border-gray-300 rounded-md py-2.5 px-3 text-sm text-gray-900 focus:outline-none focus:border-gray-500 disabled:bg-gray-100 disabled:text-gray-500"
                placeholder="you@example.com"
              />
            </div>
          </div>

          {view === 'verify_otp' && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">6-Digit OTP</label>
              <div>
                <input
                  type="text"
                  required
                  value={otp}
                  onChange={e => setOtp(e.target.value)}
                  className="w-full border border-gray-300 rounded-md py-2.5 px-3 text-sm text-gray-900 focus:outline-none focus:border-gray-500"
                  placeholder="123456"
                  maxLength={6}
                />
              </div>
            </div>
          )}

          {(view === 'login' || view === 'register' || view === 'verify_otp') && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-xs font-medium text-gray-600">
                  {view === 'verify_otp' ? 'New Password' : 'Password'}
                </label>
                {view === 'login' && (
                  <button
                    type="button"
                    onClick={() => { setError(null); setHint(null); setView('forgot_password'); }}
                    className="text-xs font-medium text-indigo-600 hover:text-indigo-800"
                  >
                    Forgot password?
                  </button>
                )}
              </div>
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
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary flex justify-center py-2.5 mt-1"
          >
            {loading ? 'Please wait...' : (
              view === 'login' ? 'Sign In' :
              view === 'register' ? 'Sign Up' :
              view === 'forgot_password' ? 'Send OTP' :
              'Reset Password'
            )}
          </button>
        </form>

        <div className="mt-5 text-center text-sm text-gray-600">
          {(view === 'login' || view === 'forgot_password' || view === 'verify_otp') && (
            <>
              Don't have an account?{' '}
              <button
                onClick={() => { setView('register'); setError(null); setHint(null); }} 
                className="text-gray-900 font-medium underline"
              >
                Sign up
              </button>
            </>
          )}
          {view === 'register' && (
            <>
              Already have an account?{' '}
              <button
                onClick={() => { setView('login'); setError(null); setHint(null); }} 
                className="text-gray-900 font-medium underline"
              >
                Sign in
              </button>
            </>
          )}
          {(view === 'forgot_password' || view === 'verify_otp') && (
            <div className="mt-2">
              <button
                onClick={() => { setView('login'); setError(null); setHint(null); }} 
                className="text-gray-500 hover:text-gray-800 font-medium underline text-xs"
              >
                Back to Sign in
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
