/**
 * Global `t()` function — installed as app.config.globalProperties.$t
 *
 * Parses bilingual text in the format "中文（English）" and returns
 * the appropriate portion based on the current language setting.
 *
 * Usage in templates:  {{ $t('用户名（Username）') }}
 * Usage in <script>:   import { t } from '@/utils/i18n'
 *                       t('用户名（Username）')
 */

let _getStore = null  // lazy reference to useSettingsStore

export function initI18n(getStoreFn) {
  _getStore = getStoreFn
}

export function t(text) {
  if (!text || !_getStore) return text
  const lang = _getStore().language

  if (lang === 'zh') {
    // Extract Chinese part before the first "（"
    const m = text.match(/^(.+?)（/)
    return m ? m[1].trim() : text
  }
  if (lang === 'en') {
    // Extract English part inside "（...）"
    const m = text.match(/（(.+?)）$/)
    return m ? m[1].trim() : text
  }
  // 'both' — return as-is
  return text
}
