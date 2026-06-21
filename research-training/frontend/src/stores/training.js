import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  startTraining as startApi,
  stopTraining as stopApi,
  getTrainingStatus,
} from '../api'

export const useTrainingStore = defineStore('training', () => {
  const currentExperimentId = ref(null)
  const isRunning = ref(false)
  const logs = ref([])
  const serverMetrics = ref([])    // global_val, global_test, best_val, best_test
  const clientMetrics = ref([])    // per-client acc/loss

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

  function reset() {
    currentExperimentId.value = null
    isRunning.value = false
    logs.value = []
    serverMetrics.value = []
    clientMetrics.value = []
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

  function addLog(line) {
    logs.value.push(line)
  }

  function addMetrics(metricsList) {
    for (const m of metricsList) {
      if (m.source === 'server' || m.source === 'best') {
        serverMetrics.value.push(m)
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
    metricsByRound,
    reset,
    start,
    stop,
    checkStatus,
    addLog,
    addMetrics,
  }
})
