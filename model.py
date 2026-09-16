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
#  教师引导的扩散式无数据生成器
#  alias: TeacherGuidedDiffusionGenerator
#
#  设计定位：
#    不依赖标准 DDPM 的噪声预测预训练（L_simple）。
#    生成器从纯高斯噪声 x_T ~ N(0, I) 出发，
#    使用扩散式多步反向变换结构，
#    由客户端教师模型的语义反馈和模型分歧进行无数据训练。
#    服务端不接触客户端原始图数据。
# =========================================================================

class ConditionalDiffusionGenerator(nn.Module):
    """
    教师引导的扩散式无数据生成器 (Teacher-Guided Diffusion-Style Data-Free Generator)

    与标准 DDPM 的关键区别：
    - 不使用真实数据 x0 做前向加噪 / 噪声预测预训练
    - 生成器从纯高斯噪声出发，经多步反向条件变换合成伪节点特征
    - 训练信号来自 S4 对抗蒸馏的语义损失、散度损失和多样性损失
    - 服务端不访问客户端原始 data.x / data.edge_index

    本类同时提供：
      differentiable_sample() —— 训练用，保留计算图
      sample()               —— 推理/验证用，torch.no_grad()
    """

    def __init__(self, feat_dim, num_classes, hidden_dim=256,
                 num_steps=50, beta_start=1e-4, beta_end=0.02,
                 output_bound='tanh'):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.output_bound = output_bound

        # Variance schedule (DDPM structure, used only for reverse step shape)
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

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
        return self.net(h)

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

    def _reverse_step(self, x_t, t, labels, num_steps_total):
        """单步反向变换 (可被 checkpoint 分组包裹)。"""
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        eps_pred = self.forward(x_t, t_tensor, labels)

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        coef1 = 1.0 / torch.sqrt(alpha_t + 1e-8)
        coef2 = beta_t / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)
        mu = coef1 * (x_t - coef2 * eps_pred)
        sigma = torch.sqrt(beta_t + 1e-8)
        return mu + sigma * noise

    def differentiable_sample(self, labels, num_steps=None, guidance_fn=None,
                              backprop_mode='full', truncate_interval=1,
                              checkpoint_segments=1):
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
        x_t = torch.randn(batch_size, self.feat_dim, device=labels.device)

        if backprop_mode == 'truncated':
            for t in reversed(range(num_steps)):
                x_prev = self._reverse_step(x_t, t, labels, num_steps)
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
                    # 逐步重计算
                    x_prev = cp.checkpoint(self._reverse_step, x_t,
                                           torch.tensor(t_start, device=labels.device),
                                           labels, num_steps, use_reentrant=False)
                    if guidance_fn is not None:
                        t_tensor = torch.full((batch_size,), t_start,
                                              device=labels.device, dtype=torch.long)
                        x_prev = guidance_fn(x_prev, t_tensor, labels)
                    x_t = x_prev
                else:
                    # 分段重计算: 一组 steps 作为一个 checkpoint 单元
                    def _chunk(x, t_hi):
                        for t in range(t_hi, t_end - 1, -1):
                            x = self._reverse_step(x, t, labels, num_steps)
                        return x
                    x_prev = cp.checkpoint(_chunk, x_t,
                                           torch.tensor(t_start, device=labels.device),
                                           use_reentrant=False)
                    x_t = x_prev
                t_start -= segments
        else:  # full
            for t in reversed(range(num_steps)):
                x_prev = self._reverse_step(x_t, t, labels, num_steps)
                if guidance_fn is not None:
                    t_tensor = torch.full((batch_size,), t, device=labels.device,
                                          dtype=torch.long)
                    x_prev = guidance_fn(x_prev, t_tensor, labels)
                x_t = x_prev

        return self._apply_output_bound(x_t)

    @torch.no_grad()
    def sample(self, labels, num_steps=None, device='cpu'):
        """
        推理用反向生成 —— @torch.no_grad()，不保留计算图。
        """
        if num_steps is None:
            num_steps = self.num_steps
        num_steps = min(num_steps, self.num_steps)

        batch_size = labels.shape[0]
        x_t = torch.randn(batch_size, self.feat_dim, device=device)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((batch_size,), t, device=device,
                                  dtype=torch.long)
            eps_pred = self.forward(x_t, t_tensor, labels)

            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            alpha_bar_t = self.alpha_bars[t]

            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            mu = (1.0 / torch.sqrt(alpha_t + 1e-8)) * (
                x_t - (beta_t / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)) * eps_pred
            )
            sigma = torch.sqrt(beta_t + 1e-8)
            x_t = mu + sigma * noise

        return self._apply_output_bound(x_t)


# 别名：更准确地反映本生成器的定位
TeacherGuidedDiffusionGenerator = ConditionalDiffusionGenerator
