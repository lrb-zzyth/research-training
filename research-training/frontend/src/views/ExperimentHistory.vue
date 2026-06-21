<template>
  <div>
    <h2>Experiment History</h2>
    <ExperimentTable @view="viewExperiment" />
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import ExperimentTable from '../components/ExperimentTable.vue'

const router = useRouter()

function viewExperiment(row) {
  router.push(`/training/monitor/${row.id}`)
}
</script>
