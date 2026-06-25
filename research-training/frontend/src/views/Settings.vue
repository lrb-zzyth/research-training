<template>
  <div>
    <h2>{{ $t('设置（Settings）') }}</h2>

    <el-card shadow="never">
      <template #header>
        <span>{{ $t('语言设置（Language Settings）') }}</span>
      </template>

      <el-form label-width="180px">
        <el-form-item :label="$t('界面语言（Display Language）')">
          <el-radio-group v-model="settings.language">
            <el-radio value="both">{{ $t('中英文同时显示（Bilingual）') }}</el-radio>
            <el-radio value="zh">{{ $t('仅中文（Chinese Only）') }}</el-radio>
            <el-radio value="en">{{ $t('仅英文（English Only）') }}</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item>
          <el-tag type="info" effect="plain">
            {{ previewText }}
          </el-tag>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" style="margin-top:16px">
      <template #header>
        <span>{{ $t('其他设置（Other Settings）') }}</span>
      </template>
      <p style="color:#999; margin:0">
        {{ $t('预留位置，后续可在此添加更多配置项（Placeholder for future settings）') }}
      </p>
    </el-card>
  </div>
</template>

<script setup>
import { computed, getCurrentInstance } from 'vue'
import { useSettingsStore } from '../stores/settings'

const settings = useSettingsStore()
const globalProperties = getCurrentInstance().appContext.config.globalProperties
const $t = globalProperties.$t

const previewText = computed(() => {
  const sample = $t('用户名（Username）')
  const sample2 = $t('学习率（Learning Rate）')
  return `预览（Preview）: "${sample}"  /  "${sample2}"`
})
</script>
