import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// Request interceptor: attach JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor: handle 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api

// ── Auth API ─────────────────────────────────────────────────────────────────
export function login(username, password) {
  return api.post('/auth/login', { username, password })
}

export function register(username, password, confirmPassword) {
  return api.post('/auth/register', {
    username,
    password,
    confirm_password: confirmPassword,
  })
}

export function getMe() {
  return api.get('/auth/me')
}

// ── Training API ─────────────────────────────────────────────────────────────
export function startTraining(params) {
  return api.post('/training/start', params)
}

export function stopTraining() {
  return api.post('/training/stop')
}

export function getTrainingStatus() {
  return api.get('/training/status')
}

// ── Experiments API ──────────────────────────────────────────────────────────
export function listExperiments(limit = 50, offset = 0) {
  return api.get('/experiments/', { params: { limit, offset } })
}

export function getExperiment(id) {
  return api.get(`/experiments/${id}`)
}

export function getExperimentLogs(id, limit = 500) {
  return api.get(`/experiments/${id}/logs`, { params: { limit } })
}

export function getExperimentMetrics(id) {
  return api.get(`/experiments/${id}/metrics`)
}
