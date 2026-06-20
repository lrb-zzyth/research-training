import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, data, return_embedding=False):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        x = self.conv2(embedding, edge_index)
        if return_embedding:
            return x, embedding
        return x

    def rep_forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv2(x, edge_index)
        return x


class FedTAD_ConGenerator(nn.Module):

    def __init__(self, noise_dim, feat_dim, out_dim, dropout):
        super(FedTAD_ConGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(out_dim, out_dim)
        
        hid_layers = []
        dims = [noise_dim+out_dim, 64, 128, 256]
        for i in range(len(dims)-1):
            d_in = dims[i]
            d_out = dims[i+1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, feat_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits


class ConditionalDiffusionGenerator(nn.Module):
    """
    Small DDPM-based conditional generator for synthetic node features.
    Generates features via reverse diffusion conditioned on class labels.
    """

    def __init__(self, feat_dim, num_classes, hidden_dim=256,
                 num_steps=50, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_steps = num_steps

        # DDPM variance schedule
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
        Predict the noise at timestep t.
        x_t:     [B, feat_dim] noisy features
        t:       [B] timestep indices (0 .. num_steps-1)
        labels:  [B] class labels
        Returns [B, feat_dim] predicted noise.
        """
        t_float = t.float() / self.num_steps
        t_emb = self.time_embed(t_float.unsqueeze(-1))          # [B, H]
        c_emb = self.class_embed(labels)                        # [B, H]
        h = torch.cat([x_t, t_emb, c_emb], dim=-1)              # [B, feat+H+H]
        return self.net(h)

    def sample(self, labels, num_steps=None, device='cpu'):
        """
        Generate features via reverse diffusion (DDPM).
        labels:    [B] class labels
        num_steps: number of reverse steps (defaults to self.num_steps)
        Returns [B, feat_dim] generated features.
        """
        if num_steps is None:
            num_steps = self.num_steps
        num_steps = min(num_steps, self.num_steps)

        batch_size = labels.shape[0]
        x_t = torch.randn((batch_size, self.feat_dim), device=device)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_pred = self.forward(x_t, t_tensor, labels)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]

            # DDPM reverse step (reparameterised — noise is non-differentiable,
            # but mu depends on model, so gradients flow correctly)
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)
            mu = coef1 * (x_t - coef2 * eps_pred)
            sigma = torch.sqrt(beta_t)

            x_t = mu + sigma * noise

        return x_t