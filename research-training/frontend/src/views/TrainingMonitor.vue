<template>
  <div>
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <h2>{{ $t('训练监控（Training Monitor）') }}</h2>
      <div>
        <el-tag :type="statusTag" size="large">{{ statusText }}</el-tag>
        <el-button v-if="store.isRunning" type="danger" @click="handleStop" :loading="stopping" style="margin-left:12px">
          {{ $t('停止训练（Stop Training）') }}
        </el-button>
      </div>
    </div>

    <el-descriptions v-if="experiment" :column="3" border size="small" style="margin:12px 0">
      <el-descriptions-item :label="$t('实验 ID（Experiment ID）')">{{ experiment.id }}</el-descriptions-item>
      <el-descriptions-item :label="$t('名称（Name）')">{{ experiment.name }}</el-descriptions-item>
      <el-descriptions-item :label="$t('状态（Status）')">
        <el-tag :type="statusTag" size="small">{{ experiment.status }}</el-tag>
      </el-descriptions-item>
      <el-descriptions-item :label="$t('最佳验证准确率（Best Val）')">{{ experiment.best_val ?? '-' }}</el-descriptions-item>
      <el-descriptions-item :label="$t('最佳测试准确率（Best Test）')">{{ experiment.best_test ?? '-' }}</el-descriptions-item>
      <el-descriptions-item :label="$t('最优轮次（Best Round）')">{{ experiment.best_round ?? '-' }}</el-descriptions-item>
    </el-descriptions>

    <el-row :gutter="16" style="margin-top:12px">
      <el-col :span="14">
        <MetricChart :metrics="store.metricsByRound" />
      </el-col>
      <el-col :span="10">
        <LogConsole :logs="store.logs" style="height:400px" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useTrainingStore } from '../stores/training'
import { getExperiment, getExperimentMetrics } from '../api'
import MetricChart from '../components/MetricChart.vue'
import LogConsole from '../components/LogConsole.vue'

const route = useRoute()
const store = useTrainingStore()

const experiment = ref(null)
const stopping = ref(false)
let ws = null

const statusTag = computed(() => {
  const map = { running: 'warning', finished: 'success', failed: 'danger', stopped: 'info', pending: 'info' }
  return map[experiment.value?.status] || 'info'
})
const statusText = computed(() => {
  const map = { running: '运行中（Running）', finished: '已完成（Finished）', failed: '失败（Failed）', stopped: '已停止（Stopped）', pending: '等待中（Pending）' }
  return map[experiment.value?.status] || experiment.value?.status || 'N/A'
})

function connectWebSocket(experimentId) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${protocol}//${location.host}/api/training/ws/${experimentId}`
  ws = new WebSocket(url)

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    if (msg.type === 'log') {
      store.addLog(msg.data)
    } else if (msg.type === 'metrics') {
      store.addMetrics(msg.data)
    } else if (msg.type === 'status') {
      if (experiment.value) {
        experiment.value.status = msg.status
      }
    }
  }

  ws.onclose = () => {
    if (experiment.value?.status === 'running') {
      setTimeout(() => {
        if (experiment.value) connectWebSocket(experiment.value.id)
      }, 3000)
    }
  }
}

async function loadExperiment(id) {
  try {
    const res = await getExperiment(id)
    experiment.value = res.data
    store.currentExperimentId = id

    const metricsRes = await getExperimentMetrics(id)
    store.serverMetrics = metricsRes.data.filter(m => m.source === 'server' || m.source === 'best')
    store.clientMetrics = metricsRes.data.filter(m => m.source.startsWith('client'))

    connectWebSocket(id)
  } catch (e) {
    ElMessage.error('加载实验失败（Failed to load experiment）')
  }
}

async function handleStop() {
  try {
    await ElMessageBox.confirm('确定要停止当前训练吗？（Are you sure you want to stop this training?）')
    stopping.value = true
    await store.stop()
    if (experiment.value) experiment.value.status = 'stopped'
    ElMessage.success('训练已停止（Training stopped）')
  } catch {
    // Canceled
  } finally {
    stopping.value = false
  }
}

onMounted(async () => {
  const id = route.params.id
  if (id) {
    await loadExperiment(id)
  } else if (store.currentExperimentId) {
    await loadExperiment(store.currentExperimentId)
  }
})

onUnmounted(() => {
  if (ws) ws.close()
})
</script>
