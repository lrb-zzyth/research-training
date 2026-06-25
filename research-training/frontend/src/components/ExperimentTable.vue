<template>
  <el-card shadow="never">
    <el-table :data="experiments" v-loading="loading" stripe style="width:100%">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="name" :label="$t('名称（Name）')" min-width="160" />
      <el-table-column prop="status" :label="$t('状态（Status）')" width="110">
        <template #default="{ row }">
          <el-tag :type="statusType(row.status)" size="small">{{ statusMap[row.status] || row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="start_time" :label="$t('开始时间（Start Time）')" width="180" />
      <el-table-column prop="end_time" :label="$t('结束时间（End Time）')" width="180" />
      <el-table-column prop="best_round" :label="$t('最优轮次（Best Round）')" width="100" />
      <el-table-column prop="best_val" :label="$t('最佳验证（Best Val）')" width="100" />
      <el-table-column prop="best_test" :label="$t('最佳测试（Best Test）')" width="100" />
      <el-table-column :label="$t('操作（Actions）')" width="80" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" size="small" @click="$emit('view', row)">{{ $t('查看（View）') }}</el-button>
        </template>
      </el-table-column>
    </el-table>
    <div style="text-align:center; margin-top:12px">
      <el-pagination
        v-model:current-page="page"
        :page-size="pageSize"
        :total="total"
        layout="prev, pager, next"
        @current-change="fetchData"
      />
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { listExperiments } from '../api'

defineEmits(['view'])

const experiments = ref([])
const loading = ref(false)
const page = ref(1)
const pageSize = 20
const total = ref(0)

const statusMap = {
  running: '运行中（Running）',
  finished: '已完成（Finished）',
  failed: '失败（Failed）',
  stopped: '已停止（Stopped）',
  pending: '等待中（Pending）',
}

function statusType(status) {
  const map = { running: 'warning', finished: 'success', failed: 'danger', stopped: 'info', pending: 'info' }
  return map[status] || 'info'
}

async function fetchData() {
  loading.value = true
  try {
    const offset = (page.value - 1) * pageSize
    const res = await listExperiments(pageSize, offset)
    experiments.value = res.data
    total.value = res.data.length < pageSize ? offset + res.data.length + pageSize : offset + pageSize + 1
  } catch {
    experiments.value = []
  } finally {
    loading.value = false
  }
}

onMounted(fetchData)
</script>
