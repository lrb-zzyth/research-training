import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { login as loginApi, getMe } from '../api'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const user = ref(null)

  const isLoggedIn = computed(() => !!token.value)

  async function login(username, password) {
    const res = await loginApi(username, password)
    token.value = res.data.access_token
    localStorage.setItem('token', res.data.access_token)
    // Fetch user info
    const meRes = await getMe()
    user.value = meRes.data
  }

  function logout() {
    token.value = ''
    user.value = null
    localStorage.removeItem('token')
  }

  async function fetchUser() {
    if (token.value) {
      try {
        const meRes = await getMe()
        user.value = meRes.data
      } catch {
        logout()
      }
    }
  }

  return { token, user, isLoggedIn, login, logout, fetchUser }
})
