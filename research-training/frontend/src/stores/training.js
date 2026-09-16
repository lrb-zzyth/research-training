import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  startTraining as startApi,
  stopTraining as stopApi,
  getTrainingStatus,
  getTrainingUpdates as getUpdatesApi,
} from '../api'

export const useTrainingStore = defineStore('training', () => {
  const currentExperimentId = ref(null)
  const isRunning = ref(false)
  const logs = ref([])
  const serverMetrics = ref([])    // global_val/global_test/best_*/generator/distillation/resource
  const clientMetrics = ref([])    // per-client losses (source: client_<id>)
  const ckrMetrics = ref([])       // CKR 权重 (source: ckr, name: ckr_c<ci>c<cls>)
  const paramUpdates = ref([])     // 参数热更新审计记录

  const metricsByRound = computed(() => {
    const map = {}
    for (const m of serverMetrics.value) {
      if (m.round == null) continue
      if (!map[m.round]) map[m.round] = {}
      map[m.round][m.metric_name] = m.metric_value
    }
    return Object.entries(map)
      .map(([round, vals]) => ({ round: Number(round), ...vals }))
      .sort((a, b) => a.round - b.round)
  })

  // 最近一轮 CKR 权重矩阵: 按 (client, class) 展开
  const latestCkrMatrix = computed(() => {
    const byRound = {}
    for (const m of ckrMetrics.value) {
      if (m.round == null) continue
      if (!byRound[m.round]) byRound[m.round] = []
      byRound[m.round].push(m)
    }
    const rounds = Object.keys(byRound).map(Number)
    if (!rounds.length) return { round: null, rows: [] }
    const last = Math.max(...rounds)
    const rows = []
    for (const m of byRound[last]) {
      const mm = /^ckr_c(\d+)c(\d+)$/.exec(m.metric_name)
      if (mm) {
        rows.push({ client: Number(mm[1]), cls: Number(mm[2]), value: m.metric_value })
      }
    }
    rows.sort((a, b) => a.client - b.client || a.cls - b.cls)
    return { round: last, rows }
  })

  function reset() {
    currentExperimentId.value = null
    isRunning.value = false
    logs.value = []
    serverMetrics.value = []
    clientMetrics.value = []
    ckrMetrics.value = []
    paramUpdates.value = []
  }

  async function start(params) {
    const res = await startApi({
      name: params.name || `Experiment ${new Date().toLocaleString()}`,
      description: params.description || '',
      parameters: params,
    })
    currentExperimentId.value = res.data.id
    isRunning.value = true
    return res.data
  }

  async function stop() {
    await stopApi()
    isRunning.value = false
  }

  async function checkStatus() {
    try {
      const res = await getTrainingStatus()
      isRunning.value = res.data.is_running
      currentExperimentId.value = res.data.experiment_id
      return res.data
    } catch {
      return { is_running: false, experiment_id: null }
    }
  }

  async function refreshUpdates(experimentId) {
    try {
      const res = await getUpdatesApi(experimentId)
      paramUpdates.value = res.data
    } catch {
      paramUpdates.value = []
    }
  }

  function addLog(line) {
    logs.value.push(line)
  }

  function addMetrics(metricsList) {
    for (const m of metricsList) {
      if (m.source === 'server' || m.source === 'best') {
        serverMetrics.value.push(m)
      } else if (m.source === 'ckr') {
        ckrMetrics.value.push(m)
      } else {
        clientMetrics.value.push(m)
      }
    }
  }

  return {
    currentExperimentId,
    isRunning,
    logs,
    serverMetrics,
    clientMetrics,
    ckrMetrics,
    paramUpdates,
    metricsByRound,
    latestCkrMatrix,
    reset,
    start,
    stop,
    checkStatus,
    refreshUpdates,
    addLog,
    addMetrics,
  }
})
