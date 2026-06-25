"""
GRADATE-style multi-view multi-scale contrastive learning module.

Re-implements the contrastive components from:
  "Graph Anomaly Detection via Multi-Scale Contrastive Learning Networks
   with Augmented View" (GRADATE)

This module operates on embeddings produced by FedTAD's local GCN encoder.
It does NOT contain a separate GCN backbone.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================================================================
#  1. Edge perturbation  (adapted from GRADATE utils.py:261-293)
# =========================================================================

def aug_random_edge_pyg(edge_index, num_nodes, drop_rate=0.2):
    """
    Randomly drop and add edges to create an augmented view.

    Mirrors GRADATE's aug_random_edge(): deletes drop_rate/2 existing edges
    and inserts the same number of new (non-existent) edges.

    Args:
        edge_index: [2, E] LongTensor, PyG COO format
        num_nodes: int, number of nodes in the graph
        drop_rate: total perturbation ratio (half drop, half add)

    Returns:
        new_edge_index: [2, E'] LongTensor on the same device as input
    """
    device = edge_index.device
    percent = drop_rate / 2.0
    num_edges = edge_index.shape[1]
    num_drop = int(num_edges * percent)
    if num_drop <= 0:
        return edge_index.clone()

    # --- existing edge set ---
    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    existing = set(edge_list)

    # --- drop ---
    drop_idx = random.sample(range(num_edges), min(num_drop, num_edges))
    remaining = [e for i, e in enumerate(edge_list) if i not in drop_idx]

    # --- add (rejection sampling to avoid O(N^2)) ---
    added = []
    attempts = 0
    max_attempts = num_drop * 20
    while len(added) < num_drop and attempts < max_attempts:
        attempts += 1
        i = random.randrange(num_nodes)
        j = random.randrange(num_nodes)
        if i != j and (i, j) not in existing:
            added.append((i, j))
            existing.add((i, j))

    new_edges = remaining + added
    new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
    return new_edge_index


# =========================================================================
#  2. RWR subgraph sampling  (adapted from GRADATE utils.py:241-258)
# =========================================================================

def _build_adj_list(edge_index, num_nodes):
    """Build an undirected adjacency list from edge_index."""
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def rwr_subgraph_sampling(edge_index, num_nodes, anchor_nodes, subgraph_size,
                          restart_prob=0.5, max_length=200):
    """
    Sample subgraphs via Random Walk with Restart (RWR).

    For each anchor node, run a random walk that occasionally restarts at
    the anchor, collecting unique visited nodes until subgraph_size context
    nodes are obtained.

    Args:
        edge_index:     [2, E] LongTensor, PyG COO format
        num_nodes:      int
        anchor_nodes:   1D LongTensor or list[int]
        subgraph_size:  number of CONTEXT nodes (excluding the anchor).
                        Final subgraph length = subgraph_size + 1.
        restart_prob:   probability of jumping back to the anchor at each step
        max_length:     maximum walk steps per anchor

    Returns:
        list of lists: each inner list has length (subgraph_size + 1),
                       last element is the anchor node.
    """
    adj_list = _build_adj_list(edge_index, num_nodes)
    total_size = subgraph_size + 1  # context + anchor

    if torch.is_tensor(anchor_nodes):
        anchor_nodes = anchor_nodes.tolist()

    results = []
    for anchor in anchor_nodes:
        visited = {anchor}
        current = anchor

        for _ in range(max_length):
            if len(visited) >= total_size:
                break

            if random.random() < restart_prob:
                current = anchor
            elif adj_list[current]:
                current = random.choice(adj_list[current])
            # else: isolated node, stay put

            if current not in visited:
                visited.add(current)

        # --- build subgraph and pad if needed ---
        subgraph = list(visited)
        if len(subgraph) < total_size:
            candidates = [n for n in range(num_nodes) if n not in visited]
            if candidates:
                needed = total_size - len(subgraph)
                subgraph.extend(random.sample(candidates, min(needed, len(candidates))))

        # Trim and reorder so anchor is last
        subgraph = list(dict.fromkeys(subgraph))  # dedup, preserve order
        subgraph = subgraph[:total_size]
        if anchor in subgraph:
            subgraph.remove(anchor)
        subgraph.append(anchor)

        results.append(subgraph)

    return results


# =========================================================================
#  3. Readout modules  (GRADATE model.py:38-53)
# =========================================================================

class AvgReadout(nn.Module):
    """Average readout over the node dimension."""
    def forward(self, seq, *args):
        # seq: [B, N, H]  ->  [B, H]
        return torch.mean(seq, dim=1)


# =========================================================================
#  4. Discriminators  (GRADATE model.py:76-124)
# =========================================================================

class ContextualDiscriminator(nn.Module):
    """
    Bilinear discriminator for node-subgraph contrast.
    Scores compatibility between a subgraph embedding and an anchor node.
    """
    def __init__(self, n_h, negsamp_round):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.negsamp_round = negsamp_round
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        """
        c:    subgraph embedding    [B, H]
        h_pl: anchor node embedding [B, H]
        Returns logits [B * (1 + negsamp_round), 1]
        """
        scs = [self.f_k(h_pl, c)]
        c_mi = c
        for _ in range(self.negsamp_round):
            # Circular shift: move last to first
            c_mi = torch.cat((c_mi[-1:], c_mi[:-1]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        return torch.cat(scs, dim=0)


class PatchDiscriminator(nn.Module):
    """
    Bilinear discriminator for node-node contrast.
    Scores compatibility between two nodes within the same subgraph.
    """
    def __init__(self, n_h, negsamp_round):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.negsamp_round = negsamp_round
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, h_ano, h_unano):
        """
        h_ano:   "other" node in subgraph  [B, H]
        h_unano: anchor node               [B, H]
        Returns logits [B * (1 + negsamp_round), 1]
        """
        scs = [self.f_k(h_unano, h_ano)]
        h_mi = h_ano
        for _ in range(self.negsamp_round):
            h_mi = torch.cat((h_mi[-1:], h_mi[:-1]), 0)
            scs.append(self.f_k(h_unano, h_mi))
        return torch.cat(scs, dim=0)


# =========================================================================
#  5. GRADATE contrastive module  (no independent GCN)
# =========================================================================

class GradateContrastiveModule(nn.Module):
    """
    Multi-view multi-scale contrastive learning on top of pre-computed
    node embeddings from FedTAD's local GCN encoder.

    This module does NOT contain a GCN backbone.  It expects:
      embeddings       – node embeddings from the original graph view
      embeddings_hat   – node embeddings from the augmented graph view
    Both must come from the SAME local_model, so gradients can flow back.

    Three losses (mirroring GRADATE run.py:158-194):
      node_subgraph_loss    – ContextualDiscriminator
      node_node_loss        – PatchDiscriminator
      subgraph_subgraph_loss – cross-view InfoNCE
    """

    def __init__(self, hidden_dim, negsamp_ratio_patch=6,
                 negsamp_ratio_context=1, beta=0.1, gamma=0.1,
                 alpha=0.1, tau=0.5):
        super().__init__()
        self.readout = AvgReadout()
        self.c_disc = ContextualDiscriminator(hidden_dim, negsamp_ratio_context)
        self.p_disc = PatchDiscriminator(hidden_dim, negsamp_ratio_patch)

        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

        # BCE losses with pos_weight for imbalanced negatives
        self.bce_context = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_patch = nn.BCEWithLogitsLoss(reduction='none')

        # Store ratios for label construction
        self.negsamp_ratio_context = negsamp_ratio_context
        self.negsamp_ratio_patch = negsamp_ratio_patch

    def _bce_labels(self, batch_size, negsamp_ratio, device):
        """Build labels for discriminator: B positives, B*neg negatives."""
        total = batch_size * (1 + negsamp_ratio)
        lbl = torch.zeros(total, 1, device=device)
        lbl[:batch_size] = 1.0
        return lbl

    def _nce_loss(self, c, c_hat, batch_size):
        """
        Cross-view subgraph-subgraph InfoNCE.
        Mirrors GRADATE run.py:161-178 exactly.
        """
        c_norm = F.normalize(c, dim=1, p=2)
        c_hat_norm = F.normalize(c_hat, dim=1, p=2)

        sim_one = torch.mm(c_norm, c_hat_norm.t())    # [B, B]  cross-view
        sim_two = torch.mm(c_norm, c_norm.t())         # [B, B]  intra-view-1
        sim_three = torch.mm(c_hat_norm, c_hat_norm.t())  # [B, B]  intra-view-2

        exp_one = torch.exp(sim_one / self.tau)
        exp_two = torch.exp(sim_two / self.tau)
        exp_three = torch.exp(sim_three / self.tau)

        # Build negative index list (GRADATE shift pattern)
        # nega_list = [B-1, 0, 1, ..., B-2]
        nega_list = list(range(batch_size - 1))
        nega_list.insert(0, batch_size - 1)

        # Sum of negative-column exps from all three matrices
        row_sum = (exp_one[:, nega_list] + exp_two[:, nega_list] + exp_three[:, nega_list])
        row_sum_diag = torch.diagonal(row_sum)  # [B]

        # Positive: diagonal of cross-view similarity
        pos = torch.diagonal(sim_one)
        pos_exp = torch.exp(pos / self.tau)

        nce = -torch.log(pos_exp / (row_sum_diag + 1e-8))
        return nce.mean()

    def forward(self, embeddings, embeddings_hat, edge_index, aug_edge_index,
                train_idx, subgraph_size=4):
        """
        Args:
            embeddings:      [N, H]   original-view node embeddings
            embeddings_hat:  [N, H]   augmented-view node embeddings
            edge_index:      [2, E]   original edges
            aug_edge_index:  [2, E']  augmented edges
            train_idx:       boolean mask [N]  or long indices
            subgraph_size:   context nodes PER subgraph (excl. anchor)

        Returns:
            total_loss: scalar Tensor with gradient flowing to local_model
        """
        device = embeddings.device

        # ---- 1. Determine anchor nodes ----
        if train_idx.dtype == torch.bool:
            anchor_nodes = torch.where(train_idx)[0]
        else:
            anchor_nodes = train_idx

        # Subsample if too many (avoid OOM for large graphs)
        max_anchors = 256
        if len(anchor_nodes) > max_anchors:
            perm = torch.randperm(len(anchor_nodes), device=device)[:max_anchors]
            anchor_nodes = anchor_nodes[perm]

        batch_size = len(anchor_nodes)
        if batch_size == 0:
            return torch.tensor(0.0, device=device)

        # ---- 2. RWR subgraph sampling (CPU) ----
        num_nodes = embeddings.shape[0]
        subgraphs = rwr_subgraph_sampling(
            edge_index, num_nodes, anchor_nodes.cpu(),
            subgraph_size, restart_prob=0.5)

        # ---- 3. Extract subgraph embeddings ----
        # sub_emb:     [B, subgraph_size+1, H]  from original view
        # sub_emb_hat: [B, subgraph_size+1, H]  from augmented view
        sub_emb_list = []
        sub_emb_hat_list = []
        for sg in subgraphs:
            sg_t = torch.tensor(sg, device=device)
            sub_emb_list.append(embeddings[sg_t].unsqueeze(0))
            sub_emb_hat_list.append(embeddings_hat[sg_t].unsqueeze(0))

        sub_emb = torch.cat(sub_emb_list, dim=0)        # [B, S+1, H]
        sub_emb_hat = torch.cat(sub_emb_hat_list, dim=0)  # [B, S+1, H]

        # ---- 4. Split: anchor vs context ----
        anchor_emb = sub_emb[:, -1, :]             # [B, H]
        anchor_emb_hat = sub_emb_hat[:, -1, :]     # [B, H]
        context_emb = self.readout(sub_emb[:, :-1, :])        # [B, H]
        context_emb_hat = self.readout(sub_emb_hat[:, :-1, :])  # [B, H]
        other_emb = sub_emb[:, 0, :]                # [B, H]  first context node
        other_emb_hat = sub_emb_hat[:, 0, :]        # [B, H]

        # ---- 5. BCE labels (must match logits shape) ----
        lbl_context = self._bce_labels(batch_size, self.negsamp_ratio_context, device)
        lbl_patch = self._bce_labels(batch_size, self.negsamp_ratio_patch, device)

        # pos_weight tensors on the correct device
        pw_context = torch.tensor([self.negsamp_ratio_context], device=device, dtype=torch.float)
        pw_patch = torch.tensor([self.negsamp_ratio_patch], device=device, dtype=torch.float)

        # ---- 6. Node-subgraph contrast ----
        logits_c = self.c_disc(context_emb, anchor_emb)         # [B*(1+nc), 1]
        logits_c_hat = self.c_disc(context_emb_hat, anchor_emb_hat)
        loss_c = (self.bce_context(logits_c, lbl_context) * pw_context).mean()
        loss_c_hat = (self.bce_context(logits_c_hat, lbl_context) * pw_context).mean()
        node_subgraph_loss = self.alpha * loss_c + (1 - self.alpha) * loss_c_hat

        # ---- 7. Node-node contrast ----
        logits_p = self.p_disc(other_emb, anchor_emb)           # [B*(1+np), 1]
        logits_p_hat = self.p_disc(other_emb_hat, anchor_emb_hat)
        loss_p = (self.bce_patch(logits_p, lbl_patch) * pw_patch).mean()
        loss_p_hat = (self.bce_patch(logits_p_hat, lbl_patch) * pw_patch).mean()
        node_node_loss = self.alpha * loss_p + (1 - self.alpha) * loss_p_hat

        # ---- 8. Subgraph-subgraph cross-view InfoNCE ----
        subgraph_loss = self._nce_loss(context_emb, context_emb_hat, batch_size)

        # ---- 9. Combine ----
        total = (self.beta * node_subgraph_loss +
                 (1 - self.beta) * node_node_loss +
                 self.gamma * subgraph_loss)
        return total
