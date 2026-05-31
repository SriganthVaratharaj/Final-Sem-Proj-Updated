import { createContext, useContext, useState, useEffect } from 'react'
import { jwtDecode } from 'jwt-decode'

const AuthContext = createContext()

export function AuthProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [user, setUser] = useState(null)
  const [isAuthModalOpen, setAuthModalOpen] = useState(false)

  useEffect(() => {
    if (token) {
      try {
        const decoded = jwtDecode(token)
        if (decoded.exp * 1000 < Date.now()) {
          logout()
        } else {
          setUser({ email: decoded.sub })
          localStorage.setItem('token', token)
        }
      } catch (err) {
        logout()
      }
    } else {
      setUser(null)
      localStorage.removeItem('token')
    }
  }, [token])

  const login = (newToken) => {
    setToken(newToken)
    setAuthModalOpen(false)
  }

  const logout = () => {
    setToken(null)
  }

  const openAuth = () => setAuthModalOpen(true)
  const closeAuth = () => setAuthModalOpen(false)

  return (
    <AuthContext.Provider value={{ token, user, login, logout, isAuthModalOpen, openAuth, closeAuth }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
