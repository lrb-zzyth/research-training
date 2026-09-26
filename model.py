import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.

    Refactored into encode() + classify_from_embedding() to support
    subgraph-level contrastive learning that shares conv1 between
    classification and contrastive tasks.
    """

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        """Shared encoder: conv1 -> ReLU -> Dropout, returns hidden embedding."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def classify_from_embedding(self, h, edge_index):
        """Classification head: conv2 -> logits."""
        return self.conv2(h, edge_index)

    def forward(self, data, return_embedding=False):
        h = self.encode(data.x, data.edge_index)
        logits = self.classify_from_embedding(h, data.edge_index)
        if return_embedding:
            return logits, h
        return logits

    def rep_forward(self, data):
        return self.conv2(data.x, data.edge_index)


# =========================================================================
#  联邦预训练条件扩散生成器 + 教师引导对抗精调
#  (Federated-Pretrained Conditional Diffusion Generator
#   with Teacher-Guided Adversarial Refinement)
#  alias: TeacherGuidedDiffusionGenerator
#
#  两阶段训练 (见 train_fedtad.py)：
#    阶段1 联邦条件 DDPM 预训练 (--federated_diffusion_pretrain, 默认开启)：
#      各客户端本地对真实 x0 做前向加噪 + 噪声预测 (denoise_loss, 专利 S3.1/S3.2)，
#      仅上传去噪网络参数，服务端按节点数加权聚合 —— 原始特征不出域 (专利权1)。
#    阶段2 教师引导对抗精调：
#      服务端以 L_sem / L_dis / L_div 训生成器（生成器最大化 L_dis 分歧），
#      从纯高斯噪声 x_T ~ N(0, I) 出发经多步反向条件变换合成伪节点特征。
#    伪特征 -> 余弦相似度 + KNN 构图 -> CKR 加权 KL 蒸馏 (build_knn_graph)。
#    服务端不接触客户端原始图数据。
# =========================================================================

class ConditionalDiffusionGenerator(nn.Module):
    """
    条件扩散伪特征生成器：联邦预训练 + 教师引导对抗精调
    (Federated-Pretrained Conditional Diffusion Generator
     with Teacher-Guided Adversarial Refinement)

    两阶段训练：
    1) 联邦条件 DDPM 预训练 (federated_diffusion_pretrain, 默认开启)：
       客户端本地对真实 x0 做前向加噪 + 噪声预测 (denoise_loss)，
       仅上传去噪网络参数，服务端按节点数加权聚合；原始特征不出域 (专利权1 / S3.1)。
    2) 教师引导对抗精调 (服务端每轮, L_sem / L_dis / L_div)：
       生成器从纯高斯噪声 x_T ~ N(0, I) 出发，经多步反向条件变换合成伪节点特征；
       最小化语义/多样性损失、最大化教师分歧损失（对抗）。
       伪特征经余弦相似度 + KNN 构图后，用于 CKR 加权 KL 蒸馏。
       服务端不访问客户端原始 data.x / data.edge_index。

    本类同时提供：
      denoise_loss()           —— 阶段1：标准 DDPM 前向加噪 + 噪声预测 (S3.1/S3.2)
      differentiable_sample()  —— 训练用，保留计算图
      sample()                 —— 推理/验证用，torch.no_grad()
    """

    def __init__(self, feat_dim, num_classes, hidden_dim=256,
                 num_steps=50, beta_start=1e-4, beta_end=0.02,
                 output_bound='tanh', residual_skip=False,
                 skip_mode='none', use_posterior_variance=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.output_bound = output_bound
        # reverse 步的噪声方差: True 用 posterior variance β̃_t (当前默认主行为),
        # False 用历史 √β_t 路径 (仅消融, 由 --use_posterior_variance 控制)
        self.use_posterior_variance = use_posterior_variance
        # 直通残差模式 (denoiser 内部架构):
        #   'none'            旧版 (无直通)
        #   'fixed'           eps_pred = x_t + body(x_t,t,c)   [= 旧 residual_skip]
        #   'timestep_scalar' eps_pred = c(t)·x_t + body, c(t)=exp(g_t) 每步一个
        #                     可学习正 scalar, 初始化 1 (完全复现 fixed 初始行为),
        #                     自动学 early-t 的放大系数, 修 identity skip 的结构性 bias
        if skip_mode == 'none' and residual_skip:
            skip_mode = 'fixed'   # 兼容旧参数
        self.skip_mode = skip_mode
        if skip_mode == 'timestep_scalar':
            self.skip_log_scale = nn.Embedding(num_steps, 1)
            nn.init.zeros_(self.skip_log_scale.weight)

        # Variance schedule (DDPM structure, used only for reverse step shape)
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # DDPM posterior variance β̃_t = β_t·(1−ᾱ_{t−1})/(1−ᾱ_t)
        # (β 很大时 √β_t 注入噪声过强, 用 β̃_t 修正; β̃_0 = 0 天然保证 t=0 无噪声)
        alpha_bars_prev = torch.cat([torch.ones_like(alpha_bars[:1]),
                                     alpha_bars[:-1]])
        posterior_variance = (betas * (1.0 - alpha_bars_prev)
                              / (1.0 - alpha_bars).clamp_min(1e-12))

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('posterior_variance', posterior_variance)

        # Sinusoidal-style time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Class-label embedding
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

        # Noise-prediction network (MLP conditioned on time & class)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x_t, t, labels):
        """
        预测时间步 t 的噪声 / 变换方向。

        Args:
            x_t:     [B, F] 当前特征状态
            t:       [B] 时间步索引 (0..num_steps-1)
            labels:  [B] 类别标签

        Returns:
            [B, F] 预测的变换方向
        """
        t_float = t.float() / self.num_steps
        t_emb = self.time_embed(t_float.unsqueeze(-1))          # [B, H]
        c_emb = self.class_embed(labels)                        # [B, H]
        h = torch.cat([x_t, t_emb, c_emb], dim=-1)              # [B, F+H+H]
        out = self.net(h)
        if self.skip_mode == 'fixed':
            out = x_t + out
        elif self.skip_mode == 'timestep_scalar':
            c_t = torch.exp(self.skip_log_scale(t))             # [B, 1], 初始化 exp(0)=1
            out = c_t * x_t + out
        return out

    def denoise_loss(self, x0, labels, t=None, reduce='mean'):
        """
        标准 DDPM 训练目标 —— 专利 S3.1 / S3.2 的前向加噪 + 噪声预测。

            S3.1 前向:  q(x_t | x_0) = N(x_t; √(ᾱ_t)·x_0, (1−ᾱ_t)·I)
            S3.2 损失:  L = E_{x_0,ε,t}[ ‖ε − ε_θ(x_t, t, c)‖² ]
                        "即预测并去除每一步所添加的噪声"

        Args:
            x0:     [B, F] **真实**节点特征 (来自客户端本地数据, 不出域)
            labels: [B]    这些节点的类别标签 (条件 c)
            t:      [B]    可选, 指定时间步; 默认均匀随机采样
        Returns:
            标量损失
        """
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)          # [B,1]
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        # 注意: 这里必须用未加 output_bound 的原始输出 —— 预测的是噪声(可正可负、量级~1),
        # 若过 tanh 会被夹死, 去噪任务无法学习
        eps_pred = self.forward(x_t, t, labels)
        return torch.nn.functional.mse_loss(eps_pred, eps, reduction=reduce)

    def _apply_output_bound(self, x):
        """应用输出边界约束，防止纯噪声起步时数值爆炸。"""
        if self.output_bound == 'tanh':
            return torch.tanh(x)
        elif self.output_bound == 'clamp':
            return torch.clamp(x, -10.0, 10.0)
        elif self.output_bound == 'none':
            return x
        else:
            return x

    def _reverse_step(self, x_t, t, labels, num_steps_total,
                      noise_generator=None, noise_override=None):
        """单步反向变换 (可被 checkpoint 分组包裹)。

        noise_generator: 可选的 torch.Generator, 固定逐步噪声 (纵向诊断用,
        每次采样前由调用方重置种子); None 时用全局 RNG (默认行为不变)。
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        eps_pred = self.forward(x_t, t_tensor, labels)

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        if t > 0:
            if noise_override is not None:
                noise = noise_override
            else:
                noise = (torch.randn(x_t.shape, generator=noise_generator,
                                     device=x_t.device, dtype=x_t.dtype)
                         if noise_generator is not None else torch.randn_like(x_t))
        else:
            noise = torch.zeros_like(x_t)
        coef1 = 1.0 / torch.sqrt(alpha_t + 1e-8)
        coef2 = beta_t / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)
        mu = coef1 * (x_t - coef2 * eps_pred)
        # posterior variance β̃_t (默认主行为) / 历史 √β_t 路径 (--no-use_posterior_variance)
        variance = (self.posterior_variance[t]
                    if self.use_posterior_variance else self.betas[t])
        sigma = torch.sqrt(variance)
        return mu + sigma * noise

    def differentiable_sample(self, labels, num_steps=None, guidance_fn=None,
                              backprop_mode='full', truncate_interval=1,
                              checkpoint_segments=1, apply_output_bound=True,
                              initial_noise=None, noise_generator=None):
        """
        可微反向生成 —— 训练用，保留计算图。

        反向传播模式 (--generator_backprop_mode):
          full:         完整计算图 (严格基线)
          checkpointed: 每 checkpoint_segments 步用 torch.utils.checkpoint
                        重计算激活, 生成器参数梯度完整 (默认推荐)
          truncated:    每隔 truncate_interval 步对状态 detach,
                        截断反向传播, 梯度不回传到更早的采样状态 (低显存近似)

        注意:
          - 初始 x_T 由 randn 生成, 不需要 requires_grad
          - 生成器参数梯度由每一步 eps_pred 对参数的依赖保证非零
          - truncated 模式与 full 模式不等价 (文档明示)

        Args:
            labels:      [B] 类别标签
            num_steps:   反向步数（默认 self.num_steps）
            guidance_fn: 可选的引导函数 fn(x_t, t, labels) -> x_t_guided
            backprop_mode: 'full' | 'checkpointed' | 'truncated'
            truncate_interval: truncated 模式每隔 N 步 detach 一次
            checkpoint_segments: checkpointed 模式每 N 步分组重计算
            apply_output_bound: False 时返回 raw 反向输出 (radius 流形约束等
                                外部参数化需要, 绕过末段 tanh)
            initial_noise: 可选固定 x_T; 用于服务端专用 RNG/配对诊断。
            noise_generator: 可选 torch.Generator, 控制每个 reverse step 的噪声。

        Returns:
            [B, F] 生成的伪节点特征 (requires_grad=True)
        """
        assert backprop_mode in ('full', 'checkpointed', 'truncated'), \
            f"unknown backprop_mode: {backprop_mode}"
        if num_steps is None:
            num_steps = self.num_steps
        num_steps = min(num_steps, self.num_steps)
        interval = max(1, int(truncate_interval))
        segments = max(1, int(checkpoint_segments))

        batch_size = labels.shape[0]
        if initial_noise is None:
            x_t = (torch.randn(batch_size, self.feat_dim,
                               generator=noise_generator, device=labels.device)
                   if noise_generator is not None
                   else torch.randn(batch_size, self.feat_dim, device=labels.device))
        else:
            x_t = initial_noise.clone()

        # Pre-materialize every reverse-step noise tensor outside checkpointed
        # regions.  This makes forward and backward recomputation use identical
        # stochastic inputs.  It also makes full/checkpointed modes consume the
        # same RNG stream under the same seed, whether a dedicated Generator is
        # supplied or the global torch RNG is used.
        reverse_noises = [None] * num_steps
        for _t in range(1, num_steps):
            reverse_noises[_t] = (
                torch.randn(x_t.shape, generator=noise_generator,
                            device=x_t.device, dtype=x_t.dtype)
                if noise_generator is not None
                else torch.randn_like(x_t))

        if backprop_mode == 'truncated':
            for t in reversed(range(num_steps)):
                x_prev = self._reverse_step(
                    x_t, t, labels, num_steps,
                    noise_override=reverse_noises[t])
                if guidance_fn is not None:
                    t_tensor = torch.full((batch_size,), t, device=labels.device,
                                          dtype=torch.long)
                    x_prev = guidance_fn(x_prev, t_tensor, labels)
                x_t = x_prev
                if t > 0:
                    done_steps = num_steps - 1 - t
                    if (done_steps + 1) % interval == 0:
                        # 截断: 丢弃到更早采样状态的梯度路径
                        x_t = x_t.detach()
        elif backprop_mode == 'checkpointed':
            import torch.utils.checkpoint as cp
            t_start = num_steps - 1
            while t_start >= 0:
                t_end = max(0, t_start - segments + 1)
                if segments <= 1:
                    # Capture Python timestep in the closure; do not route control
                    # flow through a CUDA scalar tensor during recomputation.
                    _t = t_start
                    def _one_step(x, t_fixed=_t):
                        return self._reverse_step(
                            x, t_fixed, labels, num_steps,
                            noise_override=reverse_noises[t_fixed])
                    x_prev = cp.checkpoint(
                        _one_step, x_t, use_reentrant=False,
                        preserve_rng_state=False)
                    if guidance_fn is not None:
                        t_tensor = torch.full((batch_size,), _t,
                                              device=labels.device, dtype=torch.long)
                        x_prev = guidance_fn(x_prev, t_tensor, labels)
                    x_t = x_prev
                else:
                    # Group a contiguous reverse-step segment into one checkpoint.
                    _hi, _lo = t_start, t_end
                    def _chunk(x, t_hi=_hi, t_lo=_lo):
                        for t in range(t_hi, t_lo - 1, -1):
                            x = self._reverse_step(
                                x, t, labels, num_steps,
                                noise_override=reverse_noises[t])
                        return x
                    x_prev = cp.checkpoint(
                        _chunk, x_t, use_reentrant=False,
                        preserve_rng_state=False)
                    x_t = x_prev
                t_start -= segments
        else:  # full
            for t in reversed(range(num_steps)):
                x_prev = self._reverse_step(
                    x_t, t, labels, num_steps,
                    noise_override=reverse_noises[t])
                if guidance_fn is not None:
                    t_tensor = torch.full((batch_size,), t, device=labels.device,
                                          dtype=torch.long)
                    x_prev = guidance_fn(x_prev, t_tensor, labels)
                x_t = x_prev

        return self._apply_output_bound(x_t) if apply_output_bound else x_t

    @torch.no_grad()
    def sample(self, labels, num_steps=None, device='cpu',
               apply_output_bound=True, initial_noise=None,
               noise_generator=None):
        """
        推理用反向生成 —— @torch.no_grad()，不保留计算图。
        与 differentiable_sample 共用 _reverse_step 的同一套反向公式
        (posterior variance 等修正两处自动一致, 无双实现漂移)。

        initial_noise: 可选的固定初始噪声 [B, F]。None 时随机采样
        (默认行为不变); 传固定值用于纵向诊断 (跨轮比较模型变化时,
        排除 sampling noise 干扰)。
        noise_generator: 可选的 torch.Generator, 固定逐步反向噪声。
        注意: generator 状态随调用推进, 重复采样前需由调用方重置种子。
        """
        if num_steps is None:
            num_steps = self.num_steps
        num_steps = min(num_steps, self.num_steps)

        batch_size = labels.shape[0]
        if initial_noise is None:
            x_t = torch.randn(batch_size, self.feat_dim, device=device)
        else:
            x_t = initial_noise.clone()

        for t in reversed(range(num_steps)):
            x_t = self._reverse_step(x_t, t, labels, num_steps,
                                     noise_generator=noise_generator)

        return self._apply_output_bound(x_t) if apply_output_bound else x_t


# 历史别名（仅反映阶段2「教师引导」命名），保留兼容
TeacherGuidedDiffusionGenerator = ConditionalDiffusionGenerator
