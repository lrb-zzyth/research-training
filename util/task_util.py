import hashlib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data


# =========================================================================
#  1. 基本工具函数
# =========================================================================

def accuracy(pred, ground_truth):
    y_hat = pred.max(1)[1]
    correct = (y_hat == ground_truth).nonzero().shape[0]
    acc = correct / ground_truth.shape[0]
    return acc * 100


def student_loss(s_logit, t_logit, return_t_logits=False, method="kl"):
    if method == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif method == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(method)
    if return_t_logits:
        return loss, t_logit.detach()
    return loss


class DiversityLoss(nn.Module):
    def __init__(self, metric='l1'):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


# =========================================================================
#  2. 拓扑感知节点嵌入  (T = D⁻¹A 对角元素)
# =========================================================================

def cal_topo_emb(edge_index, num_nodes, max_walk_length):
    A = to_dense_adj(add_self_loops(edge_index)[0],
                     max_num_nodes=num_nodes).squeeze()
    deg = A.sum(dim=1, keepdim=True)
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    T = A * deg_inv
    diags = []
    T_pow = T.clone()
    for _ in range(max_walk_length):
        diags.append(torch.diag(T_pow).view(-1, 1))
        T_pow = torch.mm(T_pow, T)
    return torch.cat(diags, dim=1)


# =========================================================================
#  3. 伪图构建 (kNN)
# =========================================================================

def construct_graph(node_logits, adj_logits, k=5):
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge_index, _ = dense_to_sparse(adjacency_matrix.long())
    edge_index = add_self_loops(edge_index)[0]
    return Data(x=node_logits, edge_index=edge_index)


# =========================================================================
#  4. 边扰动图增强
# =========================================================================

def edge_perturbation(edge_index, num_nodes, drop_rate=0.2):
    device = edge_index.device
    percent = drop_rate / 2.0
    num_edges = edge_index.shape[1]
    num_drop = int(num_edges * percent)
    if num_drop <= 0:
        return edge_index.clone()

    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    existing = set(edge_list)

    drop_idx = random.sample(range(num_edges), min(num_drop, num_edges))
    remaining = [e for i, e in enumerate(edge_list) if i not in drop_idx]

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
    new_edge_index = torch.tensor(new_edges, dtype=torch.long,
                                  device=device).t().contiguous()
    return new_edge_index


# =========================================================================
#  5. RWR 局部子图采样 —— 返回子图列表 + 中心节点 local index
# =========================================================================

# 邻接表按 (edge hash, num_nodes) 缓存:同一客户端的原图 (view_1) 结构固定,
# view_2 每轮扰动后重建, 避免每个锚点采样都重建 O(E) 邻接表 (性能热点)。
_ADJ_CACHE = {}
_ADJ_CACHE_MAX_ENTRIES = 128


def _build_adj_list(edge_index, num_nodes):
    key = (hashlib.md5(edge_index.cpu().numpy().tobytes()).hexdigest()[:16],
           num_nodes)
    cached = _ADJ_CACHE.get(key)
    if cached is not None:
        return cached
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    if len(_ADJ_CACHE) >= _ADJ_CACHE_MAX_ENTRIES:
        _ADJ_CACHE.clear()
    _ADJ_CACHE[key] = adj
    return adj


def rwr_subgraph_sampling(edge_index, num_nodes, anchor_nodes,
                          subgraph_size, restart_prob=0.5, max_length=200,
                          seed=None):
    """
    带重启的随机游走 (RWR) 局部子图采样。

    对每个锚点节点运行 RWR，收集最多 subgraph_size 个上下文节点。
    返回的子图列表中，锚点的位置由 center_indices[i] 给出。

    seed 非 None 时使用独立的 random.Random(seed) 实例,
    相同 seed + 相同图/参数 => 相同采样结构 (用于 RWR 缓存 key 的确定性)。

    Returns:
        subgraph_lists: list[list[int]] — 每个子图的全局节点 ID
        center_indices: list[int] — 每个子图的锚点 local index
    """
    adj_list = _build_adj_list(edge_index, num_nodes)
    total_size = subgraph_size + 1

    rng = random.Random(seed) if seed is not None else random

    if torch.is_tensor(anchor_nodes):
        anchor_nodes = anchor_nodes.tolist()

    subgraph_lists = []
    center_indices = []

    for anchor in anchor_nodes:
        visited = {anchor}
        current = anchor

        for _ in range(max_length):
            if len(visited) >= total_size:
                break
            if rng.random() < restart_prob:
                current = anchor
            elif adj_list[current]:
                current = rng.choice(adj_list[current])
            if current not in visited:
                visited.add(current)

        subgraph = list(visited)
        if len(subgraph) < total_size:
            candidates = [n for n in range(num_nodes) if n not in visited]
            if candidates:
                needed = total_size - len(subgraph)
                subgraph.extend(rng.sample(candidates,
                                           min(needed, len(candidates))))

        # Dedup, trim
        subgraph = list(dict.fromkeys(subgraph))
        subgraph = subgraph[:total_size]
        if anchor not in subgraph:
            subgraph[-1] = anchor  # fallback: ensure anchor is present
        center_idx = subgraph.index(anchor)

        subgraph_lists.append(subgraph)
        center_indices.append(center_idx)

    return subgraph_lists, center_indices


# =========================================================================
#  6. 子图-子图跨视图 InfoNCE 对比损失
# =========================================================================

def subgraph_contrastive_loss(z1, z2, tau=0.5):
    """
    子图-子图跨视图 InfoNCE 对比损失 (对齐专利 S2.5 公式)。

    正样本: (z1[i], z2[i])
    负样本: (z1[i], z2[j!=i])  跨视图不同节点

    分母为整行 softmax 归一化项, **含正样本自身**
        exp(sim(z1_i, z2_i)/tau) + sum_{j!=i} exp(sim(z1_i, z2_j)/tau)
    等价于对相似度矩阵按行做 log_softmax 后取对角。
    """
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    sim = torch.mm(z1, z2.t()) / tau          # [B, B]
    log_prob = F.log_softmax(sim, dim=1)      # 分母含 j=i 项
    return -log_prob.diag().mean()


# =========================================================================
#  7. 类别加权交叉熵
# =========================================================================

def compute_class_weights(labels, num_classes, method='inverse', beta=0.999):
    n_c = torch.bincount(labels, minlength=num_classes).float()
    if method == 'inverse':
        N = labels.shape[0]
        weight_c = N / (num_classes * n_c)
        weight_c[n_c == 0] = 0.0
    elif method == 'effective_num':
        weight_c = (1 - beta) / (1 - beta ** n_c)
        weight_c[n_c == 0] = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")
    non_zero = weight_c > 0
    if non_zero.sum() > 0:
        weight_c[non_zero] = weight_c[non_zero] / weight_c[non_zero].mean()
    return weight_c


# =========================================================================
#  8. 多分类评估指标
# =========================================================================

def multiclass_metrics(logits, y):
    num_classes = logits.shape[1]
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    accuracy_val = correct / y.shape[0] * 100.0

    conf = torch.zeros(num_classes, num_classes, device=logits.device, dtype=torch.long)
    for t, p in zip(y, pred):
        conf[t, p] += 1

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for c in range(num_classes):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision.append(prec * 100.0)
        per_class_recall.append(rec * 100.0)
        per_class_f1.append(f1 * 100.0)

    macro_f1 = sum(per_class_f1) / num_classes
    class_counts = torch.bincount(y, minlength=num_classes).float()
    total = class_counts.sum().item()
    if total > 0:
        weighted_f1 = sum(f * class_counts[c].item() / total
                          for c, f in enumerate(per_class_f1))
    else:
        weighted_f1 = 0.0

    return {
        'accuracy': accuracy_val,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': conf.tolist(),
    }


# =========================================================================
#  9. 二分类异常检测评估指标
# =========================================================================

def binary_anomaly_metrics(logits, y):
    import sklearn.metrics as skm
    prob = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    prob_pos = prob[:, 1].cpu().numpy()
    y_np = y.cpu().numpy()
    pred_np = pred.cpu().numpy()

    result = {}
    wlist = []

    # Accuracy
    result['accuracy'] = (pred == y).sum().item() / y.shape[0] * 100.0

    # Confusion matrix
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    tp = ((pred == 1) & (y == 1)).sum().item()
    result['confusion_matrix'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    result['precision'] = precision * 100.0
    result['recall'] = recall * 100.0
    result['f1'] = f1 * 100.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    result['specificity'] = specificity * 100.0
    result['balanced_accuracy'] = ((recall + specificity) / 2.0) * 100.0

    n_classes_in_y = len(torch.unique(y))
    if n_classes_in_y < 2:
        wlist.append(f"ROC-AUC: only {n_classes_in_y} class(es), NaN")
        result['roc_auc'] = float('nan')
    else:
        try:
            result['roc_auc'] = skm.roc_auc_score(y_np, prob_pos) * 100.0
        except Exception as e:
            wlist.append(f"ROC-AUC failed: {e}")
            result['roc_auc'] = float('nan')

    if n_classes_in_y < 2:
        result['pr_auc'] = float('nan')
    else:
        try:
            result['pr_auc'] = skm.average_precision_score(y_np, prob_pos) * 100.0
        except Exception as e:
            wlist.append(f"PR-AUC failed: {e}")
            result['pr_auc'] = float('nan')

    result['warnings'] = wlist
    return result
