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

    <!-- 任务状态卡 -->
    <el-card shadow="never" style="margin:12px 0">
      <el-descriptions :column="4" border size="small">
        <el-descriptions-item :label="$t('实验 ID')">{{ experiment?.id ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('名称')">{{ experiment?.name ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('当前轮次 / 总轮次')">
          {{ currentRound ?? '-' }} / {{ totalRounds ?? '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="$t('当前阶段')">{{ currentStage }}</el-descriptions-item>
        <el-descriptions-item :label="$t('已运行时间')">{{ elapsedText }}</el-descriptions-item>
        <el-descriptions-item :label="$t('最佳验证指标')">{{ experiment?.best_val ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('最佳测试指标')">{{ experiment?.best_test ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('最佳轮次')">{{ experiment?.best_round ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('git commit')">{{ experiment?.git_commit || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('最后有效轮次')">{{ experiment?.last_round ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('退出码')">{{ experiment?.exit_code ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('训练产物目录')">
          <span style="font-size:12px;">{{ experiment?.task_dir || '-' }}</span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- 失败原因 -->
    <el-alert v-if="experiment?.status === 'failed' && experiment?.error_tail" type="error" :closable="false"
      style="margin-bottom:12px;">
      <template #title>{{ $t('训练失败（Training Failed）') }}</template>
      <pre style="white-space:pre-wrap; max-height:200px; overflow:auto; font-size:12px;">{{ experiment.error_tail }}</pre>
    </el-alert>

    <!-- 图表与日志 -->
    <el-row :gutter="16">
      <el-col :span="14">
        <MetricChart :metrics="store.metricsByRound" title="全局指标（Global Metrics）" y-name="指标" />
        <MetricChart :metrics="clientMetricsByRound" mode="multi" :series="['ce_loss', 'cl_loss', 'total_loss']"
          title="客户端本地训练损失（Client Losses）" y-name="loss" style="margin-top:12px" />
        <MetricChart :metrics="store.metricsByRound" mode="multi"
          :series="['generator_loss', 'semantic_loss', 'diversity_loss', 'distillation_loss', 'fake_x_std']"
          title="生成器与蒸馏（Generator & Distillation）" y-name="loss" style="margin-top:12px" />
      </el-col>
      <el-col :span="10">
        <LogConsole :logs="store.logs" style="height:320px" />
        <el-card shadow="never" style="margin-top:12px;">
          <template #header>{{ $t('CKR 权重（最近一轮）') }}</template>
          <div v-if="ckrMatrix.rows.length" style="max-height:180px; overflow:auto;">
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
              <thead><tr>
                <th style="border:1px solid #eee; padding:4px;">{{ $t('客户端') }}</th>
                <th v-for="j in ckrClassCount" :key="j" style="border:1px solid #eee; padding:4px;">class {{ j - 1 }}</th>
              </tr></thead>
              <tbody>
                <tr v-for="ci in ckrClientIds" :key="ci">
                  <td style="border:1px solid #eee; padding:4px;">client {{ ci }}</td>
                  <td v-for="j in ckrClassCount" :key="j" style="border:1px solid #eee; padding:4px;">
                    {{ ckrValue(ci, j - 1) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else style="color:#909399; font-size:12px;">{{ $t('暂无 CKR 数据（等待第一轮完成）') }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 参数热更新 -->
    <el-card shadow="never" style="margin-top:12px;">
      <template #header>
        <span>{{ $t('参数热更新（下一轮开始生效）') }}</span>
        <span style="color:#909399; font-size:12px; margin-left:8px;">
          {{ $t('仅白名单参数可在当前轮结束后、下一轮开始前生效；其余参数需重新启动任务') }}
        </span>
      </template>
      <el-form inline size="small">
        <el-form-item :label="$t('参数')">
          <el-select v-model="hotParam" style="width:220px">
            <el-option v-for="(label, key) in HOT_PARAMS" :key="key" :label="label" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('新值')">
          <el-input-number v-model="hotValue" :step="0.01" style="width:160px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="submitHotUpdate" :loading="updating" :disabled="!store.isRunning">
            {{ $t('提交更新（下一轮生效）') }}
          </el-button>
        </el-form-item>
      </el-form>
      <el-table v-if="store.paramUpdates.length" :data="store.paramUpdates" size="small" max-height="220">
        <el-table-column prop="parameter" label="参数" width="140" />
        <el-table-column prop="old_value" label="旧值" width="90" />
        <el-table-column prop="new_value" label="新值" width="90" />
        <el-table-column prop="effective_round" label="生效轮次" width="90" />
        <el-table-column prop="status" label="状态" width="90" />
        <el-table-column prop="created_at" label="请求时间" />
        <el-table-column prop="reason" label="说明" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useTrainingStore } from '../stores/training'
import { getExperiment, getExperimentMetrics, updateTrainingParam } from '../api'
import MetricChart from '../components/MetricChart.vue'
import LogConsole from '../components/LogConsole.vue'

const route = useRoute()
const store = useTrainingStore()

const experiment = ref(null)
const stopping = ref(false)
const updating = ref(false)
const hotParam = ref('learning_rate')
const hotValue = ref(0.005)
const now = ref(Date.now())
let ws = null
let timer = null

// 热更新白名单 (与后端 config_schema.HOT_UPDATABLE 一致)
const HOT_PARAMS = {
  learning_rate: 'learning_rate（客户端学习率）',
  generator_lr: 'generator_lr（生成器学习率）',
  distill_lr: 'distill_lr（蒸馏学习率）',
  lambda_subgraph: 'lambda_subgraph（对比损失权重）',
  lambda_sem: 'lambda_sem（语义损失权重）',
  lambda_diversity: 'lambda_diversity（多样性损失权重）',
  lambda_disagreement: 'lambda_disagreement（分歧损失权重）',
  lambda_feature_norm: 'lambda_feature_norm（特征范数权重）',
  contrastive_temperature: 'contrastive_temperature（InfoNCE 温度）',
  generator_steps: 'generator_steps（生成器步数）',
  distillation_steps: 'distillation_steps（蒸馏步数）',
  edge_perturb_ratio: 'edge_perturb_ratio（边扰动比例）',
  knn_k: 'knn_k（KNN 近邻数）',
}

const statusTag = computed(() => {
  const map = { running: 'warning', finished: 'success', failed: 'danger', stopped: 'info', pending: 'info' }
  return map[experiment.value?.status] || 'info'
})
const statusText = computed(() => {
  const map = { running: '运行中（Running）', finished: '已完成（Finished）', failed: '失败（Failed）', stopped: '已停止（Stopped）', pending: '等待中（Pending）' }
  return map[experiment.value?.status] || experiment.value?.status || 'N/A'
})

// 当前轮次/总轮次
const currentRound = computed(() => {
  const rows = store.metricsByRound
  if (!rows.length) return experiment.value?.last_round ?? 0
  return Math.max(...rows.map(r => r.round))
})
const totalRounds = computed(() => experiment.value?.canonical_config?.num_rounds ?? null)

// 阶段: 从真实日志关键词推导 (展示用, 非伪造指标)
const STAGE_MARKERS = [
  ['[Client Local Training]', 'client_training'],
  ['[Dynamic CKR Update]', 'server_weights'],
  ['[Server Weights]', 'server_weights'],
  ['[Server FedAvg]', 'fedavg'],
  ['[Generator Update]', 'generator_update'],
  ['[Global Distillation]', 'global_distillation'],
  ['[Evaluation]', 'evaluation'],
  ['[Final Test]', 'final_test'],
  ['训练结束', 'finished'],
]
const currentStage = computed(() => {
  let stage = 'initializing'
  for (let i = store.logs.length - 1; i >= 0; i--) {
    const line = store.logs[i]
    for (const [marker, name] of STAGE_MARKERS) {
      if (line.includes(marker)) {
        stage = name
        break
      }
    }
    if (stage !== 'initializing') break
  }
  return stage
})

// 已运行时间
const elapsedText = computed(() => {
  if (!experiment.value?.start_time) return '-'
  const end = experiment.value.end_time ? new Date(experiment.value.end_time) : new Date(now.value)
  const start = new Date(experiment.value.start_time)
  const sec = Math.max(0, Math.floor((end - start) / 1000))
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  return `${h}h ${m}m ${s}s`
})

// 客户端损失曲线 (按 client 聚合到轮次)
const clientMetricsByRound = computed(() => {
  const map = {}
  for (const m of store.clientMetrics) {
    if (m.round == null) continue
    if (!map[m.round]) map[m.round] = { round: m.round, ce_loss: [], cl_loss: [], total_loss: [] }
    const src = m.source.startsWith('client_') ? m.source.slice(7) : '?'
    map[m.round][m.metric_name] = map[m.round][m.metric_name] || []
    if (m.metric_name in map[m.round]) {
      map[m.round][m.metric_name].push({ client: Number(src), value: m.metric_value })
    }
  }
  // 每个指标取客户端均值 (避免多条曲线过密)
  return Object.values(map)
    .sort((a, b) => a.round - b.round)
    .map(r => {
      const out = { round: r.round }
      for (const k of ['ce_loss', 'cl_loss', 'total_loss']) {
        if (r[k] && r[k].length) {
          out[k] = r[k].reduce((s, x) => s + x.value, 0) / r[k].length
        }
      }
      return out
    })
})

// CKR 矩阵
const ckrMatrix = computed(() => store.latestCkrMatrix)
const ckrClientIds = computed(() => [...new Set(ckrMatrix.value.rows.map(r => r.client))].sort((a, b) => a - b))
const ckrClassCount = computed(() => {
  const ids = ckrMatrix.value.rows.map(r => r.cls)
  return ids.length ? Math.max(...ids) + 1 : 0
})
function ckrValue(ci, cls) {
  const row = ckrMatrix.value.rows.find(r => r.client === ci && r.cls === cls)
  return row ? row.value.toFixed(3) : '-'
}

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
        if (msg.status === 'finished' || msg.status === 'failed' || msg.status === 'stopped') {
          loadExperiment(experiment.value.id)
        }
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
    store.ckrMetrics = metricsRes.data.filter(m => m.source === 'ckr')
    store.paramUpdates = []

    await store.refreshUpdates(id)
    connectWebSocket(id)
  } catch (e) {
    ElMessage.error('加载实验失败（Failed to load experiment）')
  }
}

async function submitHotUpdate() {
  if (!store.isRunning) {
    ElMessage.warning('训练未运行，无法热更新参数')
    return
  }
  updating.value = true
  try {
    const res = await updateTrainingParam(hotParam.value, hotValue.value)
    ElMessage.success(`参数 ${hotParam.value} 更新请求已提交（下一轮生效）`)
    await store.refreshUpdates(experiment.value.id)
  } catch (e) {
    const detail = e.response?.data?.detail
    const msg = typeof detail === 'string' ? detail : JSON.stringify(detail)
    ElMessage.error(msg || '热更新失败')
  } finally {
    updating.value = false
  }
}

async function handleStop() {
  try {
    await ElMessageBox.confirm('确定要停止当前训练吗？（Are you sure you want to stop this training?）')
    stopping.value = true
    await store.stop()
    if (experiment.value) {
      experiment.value.status = 'stopped'
      loadExperiment(experiment.value.id)
    }
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
  timer = setInterval(() => { now.value = Date.now() }, 1000) // 刷新已运行时间
  watch(() => store.isRunning, (running) => {
    if (!running && experiment.value) loadExperiment(experiment.value.id)
  })
})

onUnmounted(() => {
  if (ws) ws.close()
  if (timer) clearInterval(timer)
})
</script>
