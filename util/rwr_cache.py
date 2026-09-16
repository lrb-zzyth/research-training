"""
RWR Subgraph Cache

只缓存"采样结构": sampled_global_node_ids / local_edge_index / center_local_idx。
不缓存任何含计算图的张量、x_subgraph、conv 输出或 readout 表示。

视图策略:
  view_1 (原始图): 图结构固定 -> 可跨本地 epoch、跨通信轮缓存 (view1_persistent=True)
  view_2 (扰动图): 每轮边扰动不同 -> 仅在当前轮内缓存
                   (key 包含 round_idx 与 view_2 edge hash, 新一轮自动失效)

容量限制: 达到 max_entries 时 LRU 驱逐最早插入项。
可配置关闭 (enabled=False 时所有 get 返回 miss)。
"""
import hashlib
import torch


def _edge_hash(edge_index):
    """计算 edge_index 的哈希值用于缓存 key。"""
    arr = edge_index.cpu().numpy().tobytes()
    return hashlib.md5(arr).hexdigest()[:16]


class RWRSubgraphCache:
    def __init__(self, max_entries=4096, enabled=True, view1_persistent=True):
        self._cache = {}
        self._max_entries = max_entries
        self._enabled = enabled
        self._view1_persistent = view1_persistent
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self):
        return self._enabled

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self):
        return {'hits': self._hits, 'misses': self._misses,
                'hit_rate': self.hit_rate, 'entries': len(self._cache)}

    def _make_key(self, edge_hash, anchor, subgraph_size, restart_prob, seed,
                  client_id, view, round_idx):
        if view == 1 and self._view1_persistent:
            round_part = 'persist'
        elif round_idx is None:
            round_part = 'run'   # scope=run: 不按轮失效 (正确性由 edge hash 保证)
        else:
            round_part = f'r{round_idx}'
        return (f"c{client_id}_v{view}_{round_part}_{edge_hash}_"
                f"a{anchor}_s{subgraph_size}_p{restart_prob}_seed{seed}")

    def get(self, edge_index, anchor, subgraph_size, restart_prob, seed,
            client_id=0, view=1, round_idx=0):
        """
        尝试命中缓存。round_idx=None 表示不按轮区分 (scope=run)。

        Returns:
            (subgraph_list, center_local_idx) 或 None
        """
        if not self._enabled:
            self._misses += 1
            return None
        key = self._make_key(_edge_hash(edge_index), anchor, subgraph_size,
                             restart_prob, seed, client_id, view, round_idx)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    @staticmethod
    def _assert_no_grad(value):
        """防御: 缓存不允许保存含计算图的张量 (只保存采样结构)。"""
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                raise ValueError(
                    "RWRSubgraphCache 禁止缓存 requires_grad=True 的张量 "
                    "(只能缓存采样结构, 不得缓存 GNN 输出/计算图)")
            return
        if isinstance(value, (tuple, list)):
            for v in value:
                RWRSubgraphCache._assert_no_grad(v)

    def put(self, edge_index, anchor, subgraph_size, restart_prob, seed,
            value, client_id=0, view=1, round_idx=0):
        """写入缓存。value 必须只含结构 (node ids / edge_index / center idx)。"""
        if not self._enabled:
            return
        self._assert_no_grad(value)
        if len(self._cache) >= self._max_entries:
            self._cache.pop(next(iter(self._cache)))  # LRU: 最早插入项
        key = self._make_key(_edge_hash(edge_index), anchor, subgraph_size,
                             restart_prob, seed, client_id, view, round_idx)
        self._cache[key] = value

    def invalidate_round(self, round_idx):
        """使指定轮次的 view_2 缓存失效 (下一轮不再使用旧视图)。"""
        if not self._enabled:
            return
        prefix = f'_r{round_idx}_'
        keys_to_del = [k for k in self._cache if prefix in k]
        for k in keys_to_del:
            del self._cache[k]

    def clear(self):
        self._cache.clear()
        self.reset_stats()

    def reset_stats(self):
        self._hits = 0
        self._misses = 0

    def __len__(self):
        return len(self._cache)
