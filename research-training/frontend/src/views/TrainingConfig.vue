<template>
  <div>
    <h2>Training Configuration</h2>
    <el-form :model="config" label-width="200px" size="small">
      <el-divider content-position="left">Basic</el-divider>
      <el-form-item label="Experiment Name">
        <el-input v-model="config.name" placeholder="Experiment name" />
      </el-form-item>
      <el-form-item label="Root">
        <el-input v-model="config.root" placeholder="./dataset" />
      </el-form-item>
      <el-form-item label="Dataset">
        <el-select v-model="config.dataset" style="width:200px">
          <el-option label="Cora" value="Cora" />
          <el-option label="CiteSeer" value="CiteSeer" />
          <el-option label="PubMed" value="PubMed" />
        </el-select>
      </el-form-item>
      <el-form-item label="Number of Clients">
        <el-input-number v-model="config.num_clients" :min="2" :max="100" />
      </el-form-item>
      <el-form-item label="Partition">
        <el-select v-model="config.partition" style="width:200px">
          <el-option label="Louvain" value="Louvain" />
          <el-option label="Metis" value="Metis" />
        </el-select>
      </el-form-item>
      <el-form-item label="GPU ID">
        <el-input-number v-model="config.gpu_id" :min="0" :max="7" />
      </el-form-item>
      <el-form-item label="Seed">
        <el-input-number v-model="config.seed" :min="0" />
      </el-form-item>

      <el-divider content-position="left">Training</el-divider>
      <el-form-item label="Rounds">
        <el-input-number v-model="config.num_rounds" :min="1" :max="1000" />
      </el-form-item>
      <el-form-item label="Local Epochs">
        <el-input-number v-model="config.num_epochs" :min="1" :max="50" />
      </el-form-item>
      <el-form-item label="Learning Rate">
        <el-input-number v-model="config.lr" :min="1e-5" :max="1" :step="0.001" />
      </el-form-item>
      <el-form-item label="Weight Decay">
        <el-input-number v-model="config.weight_decay" :min="0" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item label="Hidden Dim">
        <el-input-number v-model="config.hid_dim" :min="16" :max="512" />
      </el-form-item>
      <el-form-item label="Dropout">
        <el-input-number v-model="config.dropout" :min="0" :max="1" :step="0.1" />
      </el-form-item>

      <el-divider content-position="left">Generator</el-divider>
      <el-form-item label="Generator Type">
        <el-select v-model="config.generator_type" style="width:200px">
          <el-option label="MLP" value="mlp" />
          <el-option label="Diffusion" value="diffusion" />
        </el-select>
      </el-form-item>
      <template v-if="config.generator_type === 'diffusion'">
        <el-form-item label="Diffusion Steps">
          <el-input-number v-model="config.diffusion_steps" :min="1" :max="200" />
        </el-form-item>
        <el-form-item label="Diffusion Hidden">
          <el-input-number v-model="config.diffusion_hidden" :min="64" :max="1024" />
        </el-form-item>
        <el-form-item label="Beta Start">
          <el-input-number v-model="config.diffusion_beta_start" :min="1e-6" :max="0.01" :step="1e-5" />
        </el-form-item>
        <el-form-item label="Beta End">
          <el-input-number v-model="config.diffusion_beta_end" :min="0.001" :max="0.5" :step="0.001" />
        </el-form-item>
      </template>
      <el-form-item label="Global Epochs">
        <el-input-number v-model="config.glb_epochs" :min="1" :max="50" />
      </el-form-item>
      <el-form-item label="Gen LR">
        <el-input-number v-model="config.lr_g" :min="1e-5" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item label="Global LR">
        <el-input-number v-model="config.lr_d" :min="1e-5" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item label="Num Gen">
        <el-input-number v-model="config.num_gen" :min="10" :max="10000" />
      </el-form-item>
      <el-form-item label="Top-K">
        <el-input-number v-model="config.topk" :min="1" :max="100" />
      </el-form-item>
      <el-form-item label="FedTAD Mode">
        <el-select v-model="config.fedtad_mode" style="width:200px">
          <el-option label="raw_distill" value="raw_distill" />
          <el-option label="rep_distill" value="rep_distill" />
        </el-select>
      </el-form-item>

      <el-divider content-position="left">Client-side Improvements</el-divider>
      <el-form-item label="Weighted CE">
        <el-switch v-model="config.use_weighted_ce" />
      </el-form-item>
      <template v-if="config.use_weighted_ce">
        <el-form-item label="Weight Method">
          <el-select v-model="config.class_weight_method" style="width:200px">
            <el-option label="effective_num" value="effective_num" />
            <el-option label="inverse" value="inverse" />
          </el-select>
        </el-form-item>
        <el-form-item label="Beta">
          <el-input-number v-model="config.beta" :min="0.9" :max="0.9999" :step="0.001" />
        </el-form-item>
      </template>
      <el-form-item label="Contrastive Loss">
        <el-switch v-model="config.use_contrastive" />
      </el-form-item>
      <template v-if="config.use_contrastive">
        <el-form-item label="Lambda CL">
          <el-input-number v-model="config.lambda_cl" :min="0" :max="10" :step="0.01" />
        </el-form-item>
        <el-form-item label="CL Temperature">
          <el-input-number v-model="config.cl_tau" :min="0.01" :max="2" :step="0.01" />
        </el-form-item>
      </template>

      <el-divider />
      <el-form-item>
        <el-button type="primary" @click="handleStart" :loading="starting" size="default">
          Start Training
        </el-button>
        <el-button @click="resetDefaults">Reset Defaults</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useTrainingStore } from '../stores/training'

const router = useRouter()
const training = useTrainingStore()
const starting = ref(false)

const defaults = {
  name: '',
  root: './dataset',
  dataset: 'Cora',
  num_clients: 10,
  partition: 'Louvain',
  gpu_id: 0,
  seed: 2024,
  num_rounds: 3,
  num_epochs: 1,
  lr: 0.01,
  weight_decay: 0.0005,
  hid_dim: 64,
  dropout: 0.5,
  generator_type: 'mlp',
  diffusion_steps: 10,
  diffusion_hidden: 256,
  diffusion_beta_start: 0.0001,
  diffusion_beta_end: 0.02,
  glb_epochs: 1,
  lr_g: 0.001,
  lr_d: 0.001,
  num_gen: 100,
  topk: 5,
  fedtad_mode: 'raw_distill',
  use_weighted_ce: false,
  class_weight_method: 'effective_num',
  beta: 0.999,
  use_contrastive: false,
  lambda_cl: 0.1,
  cl_tau: 0.5,
}

const config = reactive({ ...defaults })

function resetDefaults() {
  Object.assign(config, defaults)
}

async function handleStart() {
  starting.value = true
  try {
    const params = { ...config }
    const result = await training.start(params)
    training.reset()
    ElMessage.success(`Experiment #${result.id} started`)
    router.push(`/training/monitor/${result.id}`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || 'Failed to start training')
  } finally {
    starting.value = false
  }
}
</script>
