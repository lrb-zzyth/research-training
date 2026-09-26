import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'

const app = createApp(App)

const pinia = createPinia()
app.use(pinia)

// i18n: install global $t for templates
import { initI18n, t } from './utils/i18n'
import { useSettingsStore } from './stores/settings'
initI18n(() => useSettingsStore())

app.config.globalProperties.$t = t

app.use(router)
app.use(ElementPlus)
app.mount('#app')
