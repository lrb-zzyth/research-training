<template>
  <div>
    <h2>{{ $t('训练配置（Training Configuration）') }}</h2>

    <!-- ==================================================== -->
    <!--  B1-B4 正式方案预设模板                                 -->
    <!-- ==================================================== -->
    <el-card shadow="never" style="margin-bottom:16px;">
      <template #header>
        <span>{{ $t('正式实验方案（B1–B4 Presets）') }}</span>
      </template>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <el-button size="small" @click="applyPreset('B1')" :type="preset === 'B1' ? 'primary' : ''">
          B1 · FedAvg
        </el-button>
        <el-button size="small" @click="applyPreset('B2')" :type="preset === 'B2' ? 'primary' : ''">
          B2 · + Weighted CE
        </el-button>
        <el-button size="small" @click="applyPreset('B3')" :type="preset === 'B3' ? 'primary' : ''">
          B3 · + Subgraph Contrast
        </el-button>
        <el-button size="small" @click="applyPreset('B4')" :type="preset === 'B4' ? 'primary' : ''">
          B4 · 完整方法（默认）
        </el-button>
        <span style="margin-left:12px;color:#909399;font-size:12px;">
          {{ presetDesc }}
        </span>
      </div>
    </el-card>

    <el-form :model="config" label-width="220px" size="small">

      <!-- ==================================================== -->
      <!--  基础设置                                              -->
      <!-- ==================================================== -->
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
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('任务模式（Task Mode）')">
        <el-select v-model="config.task_mode" style="width:200px">
          <el-option label="anomaly_binary（异常检测，核心）" value="anomaly_binary" />
          <el-option label="multiclass（多分类，兼容）" value="multiclass" />
        </el-select>
      </el-form-item>
      <template v-if="config.task_mode === 'anomaly_binary'">
        <el-form-item :label="$t('正常类别（Normal Classes）')">
          <el-input v-model="config.normal_classes" style="width:200px" />
        </el-form-item>
        <el-form-item :label="$t('异常类别（Anomaly Classes）')">
          <el-input v-model="config.anomaly_classes" style="width:200px" />
        </el-form-item>
      </template>
      <el-form-item :label="$t('客户端数量（Number of Clients）')">
        <el-input-number v-model="config.num_clients" :min="2" :max="100" />
      </el-form-item>
      <el-form-item :label="$t('分区策略（Partition）')">
        <el-select v-model="config.partition" style="width:200px">
          <el-option label="Louvain（拓扑社区发现）" value="Louvain" />
          <el-option label="Metis" value="Metis" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('分区松弛量（Partition Delta）')">
        <el-input-number v-model="config.part_delta" :min="0" :max="100" />
      </el-form-item>
      <el-form-item :label="$t('GPU 编号（GPU ID）')">
        <el-input-number v-model="config.gpu_id" :min="0" :max="7" />
      </el-form-item>
      <el-form-item :label="$t('随机种子（Seed）')">
        <el-input-number v-model="config.seed" :min="0" />
      </el-form-item>
      <el-form-item :label="$t('模型种子（Model Seed，-1=跟随 Seed）')">
        <el-input-number v-model="config.model_seed" :min="-1" />
      </el-form-item>
      <el-form-item :label="$t('划分种子（Partition/Split Seed，-1=跟随 Seed）')">
        <el-input-number v-model="config.partition_seed" :min="-1" />
        <el-input-number v-model="config.split_seed" :min="-1" style="margin-left:8px" />
      </el-form-item>

      <!-- ==================================================== -->
      <!--  训练设置                                              -->
      <!-- ==================================================== -->
      <el-divider content-position="left">{{ $t('训练设置（Training）') }}</el-divider>
      <el-form-item :label="$t('联邦轮次（Rounds）')">
        <el-input-number v-model="config.federated_rounds" :min="1" :max="1000" />
      </el-form-item>
      <el-form-item :label="$t('本地训练轮数（Local Epochs）')">
        <el-input-number v-model="config.num_epochs" :min="1" :max="50" />
      </el-form-item>
      <el-form-item :label="$t('学习率（Learning Rate）')">
        <el-input-number v-model="config.learning_rate" :min="1e-5" :max="1" :step="0.001" />
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

      <!-- ==================================================== -->
      <!--  核心方法 (B1-B4)                                      -->
      <!-- ==================================================== -->
      <el-divider content-position="left">{{ $t('核心方法（Core Method，B1–B4）') }}</el-divider>
      <el-form-item :label="$t('CKR 模式（CKR Mode）')">
        <el-select v-model="config.ckr_mode" style="width:200px">
          <el-option label="static_topology（静态拓扑，核心）" value="static_topology" />
          <el-option label="hybrid_dynamic（动态，实验性）" value="hybrid_dynamic" />
          <el-option label="dynamic_only（仅动态，实验性）" value="dynamic_only" />
          <el-option label="performance_only（仅性能，实验性）" value="performance_only" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('蒸馏权重（Distill Weighting）')">
        <el-select v-model="config.distill_weighting" style="width:200px">
          <el-option label="static_ckr（B4 完整方法）" value="static_ckr" />
          <el-option label="none（B1/B2/B3 基线，跳过蒸馏）" value="none" />
          <el-option label="equal（等权蒸馏）" value="equal" />
          <el-option label="dynamic_ckr（动态 CKR，实验性）" value="dynamic_ckr" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('伪节点类别策略（Fake Class Strategy）')">
        <el-select v-model="config.fake_class_strategy" style="width:200px">
          <el-option label="balanced（类别平衡）" value="balanced" />
          <el-option label="prior（按先验）" value="prior" />
          <el-option label="reliability（按 CKR）" value="reliability" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('每轮伪节点数（Fake Node Count）')">
        <el-input-number v-model="config.fake_node_count" :min="10" :max="10000" />
      </el-form-item>
      <el-form-item :label="$t('KNN 伪图 K（KNN K）')">
        <el-input-number v-model="config.knn_k" :min="1" :max="100" />
      </el-form-item>
      <el-form-item :label="$t('生成器更新步数（Generator Steps）')">
        <el-input-number v-model="config.generator_steps" :min="1" :max="20" />
      </el-form-item>
      <el-form-item :label="$t('蒸馏步数（Distillation Steps）')">
        <el-input-number v-model="config.distillation_steps" :min="1" :max="50" />
      </el-form-item>

      <!-- ==================================================== -->
      <!--  加权交叉熵                                            -->
      <!-- ==================================================== -->
      <el-divider content-position="left">{{ $t('类别加权交叉熵（Weighted CE）') }}</el-divider>
      <el-form-item :label="$t('启用加权交叉熵（Enable Weighted CE）')">
        <el-switch v-model="config.use_weighted_ce" />
      </el-form-item>
      <template v-if="config.use_weighted_ce">
        <el-form-item :label="$t('加权方法（Weight Method）')">
          <el-select v-model="config.class_weight_method" style="width:200px">
            <el-option label="inverse（逆频率）" value="inverse" />
            <el-option label="effective_num（有效样本数）" value="effective_num" />
          </el-select>
        </el-form-item>
        <el-form-item label="Beta">
          <el-input-number v-model="config.beta" :min="0.9" :max="0.9999" :step="0.001" />
        </el-form-item>
      </template>

      <!-- ==================================================== -->
      <!--  子图-子图跨视图对比学习                                 -->
      <!-- ==================================================== -->
      <el-divider content-position="left">{{ $t('子图-子图跨视图对比学习（Subgraph Contrastive）') }}</el-divider>
      <el-form-item :label="$t('对比学习模式（Contrastive Mode）')">
        <el-select v-model="config.contrastive_mode" style="width:200px">
          <el-option label="subgraph_cross_view（子图-子图跨视图，核心）" value="subgraph_cross_view" />
          <el-option label="none（关闭）" value="none" />
        </el-select>
      </el-form-item>
      <template v-if="config.contrastive_mode !== 'none'">
        <el-form-item :label="$t('对比损失权重（Lambda Subgraph）')">
          <el-input-number v-model="config.lambda_subgraph" :min="0" :max="10" :step="0.01" />
        </el-form-item>
        <el-form-item :label="$t('InfoNCE 温度参数（Temperature）')">
          <el-input-number v-model="config.contrastive_temperature" :min="0.01" :max="2" :step="0.01" />
        </el-form-item>
        <el-form-item :label="$t('对比批大小（Contrastive Batch Size）')">
          <el-input-number v-model="config.contrastive_batch_size" :min="2" :max="512" />
        </el-form-item>
        <el-form-item :label="$t('RWR 子图采样大小（RWR Subgraph Size）')">
          <el-input-number v-model="config.rwr_subgraph_size" :min="2" :max="20" />
        </el-form-item>
        <el-form-item :label="$t('边扰动比例（Edge Perturb Ratio）')">
          <el-input-number v-model="config.edge_perturb_ratio" :min="0.01" :max="0.5" :step="0.01" />
        </el-form-item>
        <el-form-item :label="$t('RWR 重启概率（RWR Restart Prob）')">
          <el-input-number v-model="config.rwr_restart_prob" :min="0.1" :max="0.9" :step="0.05" />
        </el-form-item>
        <el-form-item :label="$t('锚点作用域（Anchor Scope）')">
          <el-select v-model="config.contrastive_anchor_scope" style="width:200px">
            <el-option label="all_nodes（全节点）" value="all_nodes" />
            <el-option label="train_nodes（训练节点）" value="train_nodes" />
          </el-select>
        </el-form-item>
      </template>

      <!-- ==================================================== -->
      <!--  扩散生成器                                             -->
      <!-- ==================================================== -->
      <el-divider content-position="left">{{ $t('教师引导扩散式生成器（Diffusion Generator）') }}</el-divider>
      <el-form-item :label="$t('扩散步数（Diffusion Steps）')">
        <el-input-number v-model="config.diffusion_steps" :min="1" :max="200" />
      </el-form-item>
      <el-form-item :label="$t('扩散隐藏层维度（Diffusion Hidden）')">
        <el-input-number v-model="config.diffusion_hidden" :min="64" :max="1024" />
      </el-form-item>
      <el-form-item :label="$t('Beta 起始（Beta Start）')">
        <el-input-number v-model="config.diffusion_beta_start" :min="1e-6" :max="0.01" :step="1e-5" />
      </el-form-item>
      <el-form-item :label="$t('Beta 终止（Beta End）')">
        <el-input-number v-model="config.diffusion_beta_end" :min="0.001" :max="0.5" :step="0.001" />
      </el-form-item>
      <el-form-item :label="$t('生成器学习率（Generator LR）')">
        <el-input-number v-model="config.generator_lr" :min="1e-5" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item :label="$t('蒸馏学习率（Distill LR）')">
        <el-input-number v-model="config.distill_lr" :min="1e-5" :max="1" :step="0.0001" />
      </el-form-item>
      <el-form-item :label="$t('语义损失权重（Lambda Sem）')">
        <el-input-number v-model="config.lambda_sem" :min="0" :max="10" :step="0.1" />
      </el-form-item>
      <el-form-item :label="$t('多样性损失权重（Lambda Diversity）')">
        <el-input-number v-model="config.lambda_diversity" :min="0" :max="10" :step="0.1" />
      </el-form-item>

      <!-- ==================================================== -->
      <!--  双重终止机制                                           -->
      <!-- ==================================================== -->
      <el-divider content-position="left">{{ $t('双重终止机制（Double Termination）') }}</el-divider>
      <el-form-item :label="$t('F1 连续提升阈值（F1 Threshold %）')">
        <el-input-number v-model="config.f1_threshold" :min="0.01" :max="5" :step="0.01" />
        <span style="margin-left:8px;color:#909399;">{{ $t('连续三轮低于此值则正常收敛') }}</span>
      </el-form-item>
      <el-form-item :label="$t('低资源 AUC 阈值（AUC Threshold %）')">
        <el-input-number v-model="config.auc_threshold" :min="0.01" :max="10" :step="0.01" />
        <span style="margin-left:8px;color:#909399;">{{ $t('10% 尾部客户端连续两轮增幅低于此值则提前终止') }}</span>
      </el-form-item>

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
import { reactive, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useTrainingStore } from '../stores/training'

const router = useRouter()
const training = useTrainingStore()
const starting = ref(false)

// 默认核心配置等价于:
// --task_mode anomaly_binary --contrastive_mode subgraph_cross_view
// --contrastive_anchor_scope all_nodes --ckr_mode static_topology
// --fake_class_strategy balanced --distill_weighting static_ckr
// --fairness_mode none --generator_init scratch --dynamic_ckr false
const defaults = {
  // 基础
  name: '',
  root: './dataset',
  dataset: 'Cora',
  task_mode: 'anomaly_binary',
  normal_classes: '0,1,2,3',
  anomaly_classes: '4,5,6',
  num_clients: 10,
  partition: 'Louvain',
  part_delta: 20,
  gpu_id: 0,
  seed: 2024,
  model_seed: -1,
  partition_seed: -1,
  split_seed: -1,
  // 训练
  federated_rounds: 100,
  num_epochs: 3,
  learning_rate: 0.01,
  weight_decay: 0.0005,
  hid_dim: 64,
  dropout: 0.5,
  // 核心方法 (B1-B4)
  ckr_mode: 'static_topology',
  distill_weighting: 'static_ckr',
  fake_class_strategy: 'balanced',
  fake_node_count: 100,
  knn_k: 5,
  generator_steps: 1,
  distillation_steps: 5,
  // 加权 CE
  use_weighted_ce: true,
  class_weight_method: 'inverse',
  beta: 0.999,
  // 子图-子图跨视图对比
  contrastive_mode: 'subgraph_cross_view',
  edge_perturb_ratio: 0.2,
  rwr_restart_prob: 0.5,
  rwr_subgraph_size: 5,
  contrastive_batch_size: 64,
  contrastive_temperature: 0.5,
  lambda_subgraph: 0.1,
  contrastive_anchor_scope: 'all_nodes',
  // 教师引导扩散式生成器
  diffusion_steps: 10,
  diffusion_hidden: 256,
  diffusion_beta_start: 0.0001,
  diffusion_beta_end: 0.02,
  generator_lr: 0.001,
  distill_lr: 0.001,
  lambda_sem: 1.0,
  lambda_diversity: 0.1,
  // 双重终止
  f1_threshold: 0.1,
  auc_threshold: 1.0,
}

const config = reactive({ ...defaults })

// B1-B4 预设模板: 只组合既有字段, 不引入新参数
const PRESETS = {
  B1: {
    desc: 'FedAvg：普通交叉熵，无对比、无 CKR 蒸馏',
    patch: { use_weighted_ce: false, contrastive_mode: 'none', distill_weighting: 'none' },
  },
  B2: {
    desc: 'FedAvg + 类别加权交叉熵，无对比、无蒸馏',
    patch: { use_weighted_ce: true, contrastive_mode: 'none', distill_weighting: 'none' },
  },
  B3: {
    desc: 'FedAvg + Weighted CE + 子图-子图跨视图对比，无蒸馏',
    patch: { use_weighted_ce: true, contrastive_mode: 'subgraph_cross_view', distill_weighting: 'none' },
  },
  B4: {
    desc: '完整方法：Weighted CE + 子图对比 + 静态拓扑 CKR + 教师引导扩散式伪图蒸馏',
    patch: { use_weighted_ce: true, contrastive_mode: 'subgraph_cross_view',
             distill_weighting: 'static_ckr', ckr_mode: 'static_topology' },
  },
}

const preset = ref('B4')
const presetDesc = computed(() => PRESETS[preset.value]?.desc || '')

function applyPreset(name) {
  preset.value = name
  Object.assign(config, PRESETS[name].patch)
}

function resetDefaults() {
  Object.assign(config, defaults)
  preset.value = 'B4'
}

async function handleStart() {
  starting.value = true
  try {
    const params = { ...config }
    // Remove name from params (frontend only)
    delete params.name
    const result = await training.start({ name: config.name, description: '', parameters: params })
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
