import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useSettingsStore = defineStore('settings', () => {
  // 'zh' | 'en' | 'both'
  const language = ref(localStorage.getItem('fedtad_language') || 'both')

  watch(language, (val) => {
    localStorage.setItem('fedtad_language', val)
  })

  function setLanguage(mode) {
    language.value = mode
  }

  return { language, setLanguage }
})
