import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || ''
const BASE = `${API_URL}/api`

export const getFileUrl = (url) => {
  if (!url) return ''
  if (url.startsWith('http')) return url
  return `${API_URL}${url}`
}

export const uploadFiles = async (files, token = null) => {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  
  const headers = { 'Content-Type': 'multipart/form-data' }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const { data } = await axios.post(`${BASE}/upload`, form, { headers })
  return data
}

export const processFiles = async (files) => {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  const { data } = await axios.post(`${BASE}/process`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const fetchHistory = async (limit = 20) => {
  const { data } = await axios.get(`${BASE}/results?limit=${limit}`)
  return data
}

export const deleteHistoryItem = async (id) => {
  await axios.delete(`${BASE}/results/${id}`)
}

export const getStreamUrl = (jobId) => `${BASE}/stream/${jobId}`
