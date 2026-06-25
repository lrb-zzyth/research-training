<template>
  <div>
    <h2>{{ $t('训练配置（Training Configuration）') }}</h2>
    <el-form :model="config" label-width="220px" size="small">
      <el-divider content-position="left">{{ $t('基础设置（Basic）') }}</el-divider>
      <el-form-item :label="$t('实验名称（Experiment Name）')">
        <el-input v-model="config.name" :placeholder="$t('实验名称')" />
      </el-form-item>
      <el-form-item :label="$t('数据根目录（Root）')">
        <el-input v-model="config.root" placeholder="./dataset" />
      </el-form-item>
      <el-form-item :label="$t('数据集（Dataset）')">
        <el-select v-model="config.dataset" style="width:200px">
          <el-option label="Cora" value="Cora" />
          <el-option label="CiteSeer" value="CiteSeer" />
          <el-option label="PubMed" value="PubMed" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('客户端数量（Number of Clients）')">
        <el-input-number v-model="config.num_clients" :min="2" :max="100" />
      </el-form-item>
      <el-form-item :label="$t('分区策略（Partition）')">
        <el-select v-model="config.partition" style="width:200px">
          <el-option label="Louvain" value="Louvain" />
          <el-option label="Metis" value="Metis" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('GPU 编号（GPU ID）')">
        <el-input-number v-model="config.gpu_id" :min="0" :max="7" />
      </el-form-item>
      <el-form-item :label="$t('随机种子（Seed）')">
        <el-input-number v-model="config.seed" :min="0" />
      </el-form-item>

      <el-divider content-position="left">{{ $t('训练设置（Training）') }}</el-divider>
      <el-form-item :label="$t('通信轮数（Rounds）')">
        <el-input-number v-model="config.num_rounds" :min="1" :max="1000" />
      </el-form-item>
      <el-form-item :label="$t('本地训练轮数（Local Epochs）')">
        <el-input-number v-model="config.num_epochs" :min="1" :max="50" />
      </el-form-item>
      <el-form-item :label="$t('学习率（Learning Rate）')">
        <el-input-number v-model="config.lr" :min="1e-5" :max="1" :step="0.001" />
      </el-form-item>
      <el-form-item :label="$t('权重衰减（Weight Decay）')">
        <el-input-number v-model="config.weight_decay" :min="0" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item :label="$t('隐藏层维度（Hidden Dim）')">
        <el-input-number v-model="config.hid_dim" :min="16" :max="512" />
      </el-form-item>
      <el-form-item :label="$t('丢弃率（Dropout）')">
        <el-input-number v-model="config.dropout" :min="0" :max="1" :step="0.1" />
      </el-form-item>

      <el-divider content-position="left">{{ $t('生成器（Generator）') }}</el-divider>
      <el-form-item :label="$t('生成器类型（Generator Type）')">
        <el-select v-model="config.generator_type" style="width:200px">
          <el-option label="MLP" value="mlp" />
          <el-option :label="$t('扩散模型（Diffusion）')" value="diffusion" />
        </el-select>
      </el-form-item>
      <template v-if="config.generator_type === 'diffusion'">
        <el-form-item :label="$t('扩散步数（Diffusion Steps）')">
          <el-input-number v-model="config.diffusion_steps" :min="1" :max="200" />
        </el-form-item>
        <el-form-item :label="$t('扩散隐藏层（Diffusion Hidden）')">
          <el-input-number v-model="config.diffusion_hidden" :min="64" :max="1024" />
        </el-form-item>
        <el-form-item :label="$t('Beta 起始（Beta Start）')">
          <el-input-number v-model="config.diffusion_beta_start" :min="1e-6" :max="0.01" :step="1e-5" />
        </el-form-item>
        <el-form-item :label="$t('Beta 终止（Beta End）')">
          <el-input-number v-model="config.diffusion_beta_end" :min="0.001" :max="0.5" :step="0.001" />
        </el-form-item>
      </template>
      <el-form-item :label="$t('全局蒸馏轮数（Global Epochs）')">
        <el-input-number v-model="config.glb_epochs" :min="1" :max="50" />
      </el-form-item>
      <el-form-item :label="$t('生成器学习率（Gen LR）')">
        <el-input-number v-model="config.lr_g" :min="1e-5" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item :label="$t('全局模型学习率（Global LR）')">
        <el-input-number v-model="config.lr_d" :min="1e-5" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item :label="$t('生成节点数（Num Gen）')">
        <el-input-number v-model="config.num_gen" :min="10" :max="10000" />
      </el-form-item>
      <el-form-item :label="$t('KNN 拓扑 K（Top-K）')">
        <el-input-number v-model="config.topk" :min="1" :max="100" />
      </el-form-item>
      <el-form-item :label="$t('蒸馏模式（FedTAD Mode）')">
        <el-select v-model="config.fedtad_mode" style="width:200px">
          <el-option label="raw_distill" value="raw_distill" />
          <el-option label="rep_distill" value="rep_distill" />
        </el-select>
      </el-form-item>

      <el-divider content-position="left">{{ $t('客户端改进（Client-side Improvements）') }}</el-divider>
      <el-form-item :label="$t('加权交叉熵（Weighted CE）')">
        <el-switch v-model="config.use_weighted_ce" />
      </el-form-item>
      <template v-if="config.use_weighted_ce">
        <el-form-item :label="$t('加权方法（Weight Method）')">
          <el-select v-model="config.class_weight_method" style="width:200px">
            <el-option label="effective_num" value="effective_num" />
            <el-option label="inverse" value="inverse" />
          </el-select>
        </el-form-item>
        <el-form-item label="Beta（effective number）">
          <el-input-number v-model="config.beta" :min="0.9" :max="0.9999" :step="0.001" />
        </el-form-item>
      </template>

      <el-form-item :label="$t('对比学习（Contrastive Loss）')">
        <el-switch v-model="config.use_contrastive" />
      </el-form-item>
      <template v-if="config.use_contrastive">
        <el-form-item :label="$t('对比学习类型（Contrastive Type）')">
          <el-select v-model="config.contrastive_type" style="width:200px">
            <el-option :label="$t('SupCon（监督式 / Supervised）')" value="supcon" />
            <el-option :label="$t('GRADATE（自监督 / Self-supervised）')" value="gradate" />
          </el-select>
        </el-form-item>
        <el-form-item :label="$t('对比损失权重（Lambda CL）')">
          <el-input-number v-model="config.lambda_cl" :min="0" :max="10" :step="0.01" />
        </el-form-item>
        <template v-if="config.contrastive_type === 'supcon'">
          <el-form-item :label="$t('温度参数（CL Temperature / Tau）')">
            <el-input-number v-model="config.cl_tau" :min="0.01" :max="2" :step="0.01" />
          </el-form-item>
        </template>
        <template v-if="config.contrastive_type === 'gradate'">
          <el-form-item :label="$t('InfoNCE 温度（GRADATE Tau）')">
            <el-input-number v-model="config.gradate_tau" :min="0.01" :max="2" :step="0.01" />
          </el-form-item>
          <el-form-item :label="$t('边增删比例（Edge Drop Rate）')">
            <el-input-number v-model="config.gradate_edge_drop_rate" :min="0.01" :max="0.5" :step="0.01" />
          </el-form-item>
          <el-form-item :label="$t('子图大小（Subgraph Size）')">
            <el-input-number v-model="config.gradate_subgraph_size" :min="2" :max="20" />
          </el-form-item>
          <el-form-item :label="$t('权重 Beta（Node-Subgraph vs Node-Node）')">
            <el-input-number v-model="config.gradate_beta" :min="0" :max="1" :step="0.01" />
          </el-form-item>
          <el-form-item :label="$t('权重 Gamma（Subgraph-Subgraph）')">
            <el-input-number v-model="config.gradate_gamma" :min="0" :max="1" :step="0.01" />
          </el-form-item>
          <el-form-item :label="$t('权重 Alpha（原始 View vs 增强 View）')">
            <el-input-number v-model="config.gradate_alpha" :min="0" :max="1" :step="0.01" />
          </el-form-item>
          <el-form-item :label="$t('负采样比 / 节点（Neg Ratio Patch）')">
            <el-input-number v-model="config.gradate_negsamp_ratio_patch" :min="1" :max="20" />
          </el-form-item>
          <el-form-item :label="$t('负采样比 / 子图（Neg Ratio Context）')">
            <el-input-number v-model="config.gradate_negsamp_ratio_context" :min="1" :max="10" />
          </el-form-item>
        </template>
      </template>

      <el-divider />
      <el-form-item>
        <el-button type="primary" @click="handleStart" :loading="starting" size="default">
          {{ $t('开始训练（Start Training）') }}
        </el-button>
        <el-button @click="resetDefaults">{{ $t('重置（Reset Defaults）') }}</el-button>
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
  contrastive_type: 'supcon',
  lambda_cl: 0.1,
  cl_tau: 0.5,
  gradate_tau: 0.5,
  gradate_edge_drop_rate: 0.2,
  gradate_beta: 0.1,
  gradate_gamma: 0.1,
  gradate_alpha: 0.1,
  gradate_subgraph_size: 4,
  gradate_negsamp_ratio_patch: 6,
  gradate_negsamp_ratio_context: 1,
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
    ElMessage.success(`实验 #${result.id} 已启动（Experiment started）`)
    router.push(`/training/monitor/${result.id}`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '启动失败（Failed to start training）')
  } finally {
    starting.value = false
  }
}
</script>
