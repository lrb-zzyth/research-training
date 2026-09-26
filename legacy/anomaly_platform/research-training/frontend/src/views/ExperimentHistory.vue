<template>
  <div>
    <h2>{{ $t('实验历史（Experiment History）') }}</h2>
    <ExperimentTable @view="viewExperiment" />
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import ExperimentTable from '../components/ExperimentTable.vue'

const router = useRouter()

function viewExperiment(row) {
  router.push(`/training/monitor/${row.id}`)
}
</script>
