<template>
  <el-card shadow="never" style="height:100%; display:flex; flex-direction:column;">
    <template #header>
      <span>{{ $t('训练日志（Training Logs）') }}</span>
    </template>
    <div ref="logContainer" class="log-container">
      <div v-for="(log, i) in logs" :key="i" class="log-line">{{ log }}</div>
      <div v-if="logs.length === 0" class="log-empty">{{ $t('等待日志...（Waiting for logs...）') }}</div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  logs: { type: Array, default: () => [] },
})

const logContainer = ref(null)

watch(() => props.logs.length, async () => {
  await nextTick()
  if (logContainer.value) {
    logContainer.value.scrollTop = logContainer.value.scrollHeight
  }
})
</script>

<style scoped>
.log-container {
  flex: 1;
  overflow-y: auto;
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 12px;
  padding: 8px;
  border-radius: 4px;
  min-height: 300px;
}
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.5;
}
.log-empty {
  color: #888;
  text-align: center;
  padding: 40px;
}
</style>
