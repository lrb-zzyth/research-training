<template>
  <el-card shadow="never">
    <template #header>
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span>{{ title }}</span>
        <el-radio-group v-if="mode === 'global' || mode === 'best'" v-model="chartType" size="small">
          <el-radio-button value="global">global_val / global_test</el-radio-button>
          <el-radio-button value="best">best_val / best_test</el-radio-button>
        </el-radio-group>
      </div>
    </template>
    <v-chart :option="chartOption" style="height:300px" autoresize />
  </el-card>
</template>

<script setup>
import { ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent])

// mode:
//   'global' 按轮次画 global_val/global_test 与 best_val/best_test (可选切换)
//   'multi'  按轮次画 series 中指定的指标名曲线
const props = defineProps({
  metrics: { type: Array, default: () => [] },   // 按轮次聚合的行
  mode: { type: String, default: 'global' },
  series: { type: Array, default: () => [] },    // mode='multi': 指标名列表
  yName: { type: String, default: '' },
  title: { type: String, default: '指标（Metrics）' },
})

const chartType = ref('global')

const chartOption = computed(() => {
  const rounds = props.metrics.map(m => m.round)
  let series = []
  if (props.mode === 'multi') {
    series = props.series.map(name => ({
      name,
      type: 'line',
      data: props.metrics.map(m => (m[name] == null ? null : m[name])),
      smooth: true,
    }))
  } else if (chartType.value === 'global') {
    series = [
      { name: 'global_val', type: 'line', data: props.metrics.map(m => m.global_val), smooth: true },
      { name: 'global_test', type: 'line', data: props.metrics.map(m => m.global_test), smooth: true },
    ]
  } else {
    series = [
      { name: 'best_val', type: 'line', data: props.metrics.map(m => m.best_val), smooth: true },
      { name: 'best_test', type: 'line', data: props.metrics.map(m => m.best_test), smooth: true },
    ]
  }
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: series.map(s => s.name) },
    grid: { left: 60, right: 20, bottom: 30, top: 40 },
    xAxis: { type: 'category', data: rounds, name: '轮次（Round）' },
    yAxis: { type: 'value', name: props.yName },
    series,
  }
})
</script>
