import os
import os.path as osp
import re
import time
import warnings
import torch
from torch_geometric.data import Dataset
from util.base_data_util import data_partition

warnings.filterwarnings('ignore')


class DatasetBlockedError(RuntimeError):
    """
    外部资源阻塞 (网络不可达/缓存缺失), 与代码失败 (failed_code) 区分。
    训练脚本捕获后打印 [BLOCKED] 哨兵并以退出码 3 结束,
    run_experiments 记录为 external_blocked。
    """

class FGLDataset(Dataset):
    def __init__(
        self,
        args,
        root,
        name,
        num_clients,
        partition,
        train,
        val,
        test,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        part_delta=20
    ):
        start = time.time()
        self.args = args
        self.root = root
        self.name = name
        self.num_clients = num_clients
        self.partition = partition
        self.train = train
        self.val = val
        self.test = test
        self.part_delta = part_delta
        # 受控支持度划分 (anomaly_enriched) 参数: 必须从 args 复制到 self,
        # processed_dir/process 依赖这些属性 (缺失时会静默回退 natural)
        self.partition_support_mode = getattr(args, 'partition_support_mode', 'natural')
        self.partition_anomaly_classes = getattr(args, 'partition_anomaly_classes', None)
        self.partition_anomaly_target = getattr(args, 'partition_anomaly_target', 0)
        self.partition_rebalance_seed = getattr(args, 'partition_rebalance_seed', 0)
        # 阶段三: 四类随机种子拆分
        self.partition_seed = getattr(args, 'partition_seed', None)
        self.allocation_seed = getattr(args, 'allocation_seed', None)
        # 数据来源 (离线导入): auto / pyg_cache / local_raw / local_processed
        self.dataset_source = getattr(args, 'dataset_source', 'auto')
        self.offline = getattr(args, 'offline', False)
        # partition_mode: natural(缓存) / build(按 partition_seed 重建, 供 frozen split) /
        #                 frozen(只加载原始图, 子图由 frozen split 重建)
        self.partition_mode = getattr(args, 'partition_mode', 'natural')

        super(FGLDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.load_data()

        end = time.time()
        print(f"load FGL dataset {name} done ({end-start:.2f} sec)")

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        fmt_name = re.sub("-", "_", self.name)
        # 受控支持度划分使用独立缓存目录, 不覆盖 natural 划分结果
        partition_dir = self.partition
        if getattr(self, 'partition_support_mode', 'natural') == 'anomaly_enriched':
            partition_dir = f"{self.partition}_enriched"
        # frozen-split 构建模式: 缓存目录含 partition/allocation seed,
        # 保证不同 partition_seed 的 Louvain 划分互不串用
        if getattr(self, 'partition_mode', 'natural') == 'build':
            p_seed = getattr(self, 'partition_seed', 0) or 0
            a_seed = getattr(self, 'allocation_seed', 0) or 0
            partition_dir = f"{partition_dir}_frozen_p{p_seed}_a{a_seed}"
        return osp.join(
            self.raw_dir, fmt_name, "Client{}".format(
                self.num_clients), partition_dir
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ["data{}.pt".format(i) for i in range(self.num_clients)]
        return files_names

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data{}.pt".format(idx)),
            weights_only=False)
        return data

    def process(self):
        self.load_global_graph()

        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        subgraph_list, node_dict = data_partition(
            G=self.global_data,
            num_clients=self.num_clients,
            train=self.train,
            val=self.val,
            test=self.test,
            partition=self.partition,
            part_delta=self.part_delta,
            support_mode=getattr(self, 'partition_support_mode', 'natural'),
            anomaly_classes=getattr(self, 'partition_anomaly_classes', None),
            anomaly_partition_target=getattr(self, 'partition_anomaly_target', 0),
            support_rebalance_seed=getattr(self, 'partition_rebalance_seed', 0),
            partition_seed=getattr(self, 'partition_seed', None),
            return_node_dict=True,
        )

        for i in range(self.num_clients):
            torch.save(subgraph_list[i], self.processed_paths[i])

        # 持久化社区移动记录 (受控支持度划分)
        if getattr(self, 'partition_support_mode', 'natural') == 'anomaly_enriched':
            moves = getattr(self.global_data, 'partition_moves', [])
            import json
            with open(osp.join(self.processed_dir, 'partition_moves.json'), 'w') as f:
                json.dump(moves, f, indent=1)
        # 持久化节点分配 (全局节点 ID -> 客户端), frozen split artifact 依赖
        import json as _json
        with open(osp.join(self.processed_dir, 'node_assignments.json'), 'w') as f:
            _json.dump({str(k): v for k, v in node_dict.items()}, f)

    def _blocked(self, reason):
        raise DatasetBlockedError(
            f"external resource blocked for dataset {self.name}: {reason}. "
            f"source={self.dataset_source} offline={self.offline}")

    def load_global_graph(self):
        source = self.dataset_source
        if source == 'local_raw':
            # 从 dataset_root 加载原始数据 (不联网)
            try:
                self._load_local_raw()
            except DatasetBlockedError:
                raise
            except Exception as e:
                self._blocked(f"local_raw 加载失败: {e}")
            return
        if source == 'local_processed':
            try:
                self._load_local_processed()
            except DatasetBlockedError:
                raise
            except Exception as e:
                self._blocked(f"local_processed 加载失败: {e}")
            return
        # auto / pyg_cache
        if self.offline or source == 'pyg_cache':
            if self._pyg_cache_exists():
                self._load_pyg()
                return
            self._blocked(f"PyG 缓存缺失 (offline={self.offline})")
        # auto: 尝试加载, 失败视为外部阻塞 (不是代码失败)
        try:
            self._load_pyg()
        except Exception as e:
            msg = str(e)
            if any(k in msg.lower() for k in ('connect', 'download', 'timeout',
                                              'ssl', 'github')):
                self._blocked(f"下载失败: {msg[:200]}")
            raise
        if self.offline:
            self._blocked("offline 模式下不应走到这里")

    def _pyg_cache_exists(self):
        fmt_name = re.sub("-", "_", self.name)
        raw_dir = osp.join(self.raw_dir, fmt_name, 'raw')
        return osp.exists(raw_dir) and len(os.listdir(raw_dir)) > 0

    def _load_pyg(self):
        if self.name in ["Cora", "CiteSeer", "PubMed"]:
            from torch_geometric.datasets import Planetoid
            self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)
        elif self.name in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            self.global_dataset = PygNodePropPredDataset(
                root=self.raw_dir, name=self.name
            )
        elif self.name in ["CS", "Physics"]:
            from torch_geometric.datasets import Coauthor
            self.global_dataset = Coauthor(root=self.raw_dir, name=self.name)
        elif self.name in ["Computers", "Photo"]:
            from torch_geometric.datasets import Amazon
            self.global_dataset = Amazon(
                root=self.raw_dir, name=self.name.lower())
        elif self.name in ["NELL"]:
            from torch_geometric.datasets import NELL
            self.global_dataset = NELL(
                root=os.path.join(self.raw_dir, name="NELL"))
        elif self.name in ["Reddit"]:
            from torch_geometric.datasets import Reddit
            self.global_dataset = Reddit(
                root=os.path.join(self.raw_dir, name="Reddit"))
        elif self.name in ["Flickr"]:
            from torch_geometric.datasets import Flickr
            self.global_dataset = Flickr(
                root=os.path.join(self.raw_dir, name="Flickr"))
        else:
            raise ValueError(
                "Not supported for this dataset, please check root file path and dataset name"
            )
        self.global_data = self.global_dataset.data
        self.global_data.num_classes = self.global_dataset.num_classes

    def _load_local_raw(self):
        """local_raw: 从 dataset_root 加载原始数据 (不联网, 不生成替代数据)。"""
        from torch_geometric.datasets import Planetoid
        self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)
        self.global_data = self.global_dataset.data
        self.global_data.num_classes = self.global_dataset.num_classes

    def _load_local_processed(self):
        """local_processed: 读取已处理的 .pt Data 文件 (含校验)。"""
        import glob as _glob
        raw_dir = self.raw_dir
        pt_files = sorted(_glob.glob(osp.join(raw_dir, '**', '*.pt'),
                                     recursive=True))
        if not pt_files:
            self._blocked(f"local_processed: {raw_dir} 下无 .pt 文件")
        # 取规模最大的 pt 作为全局图 (校验特征维度/标签数)
        best = None
        for f in pt_files:
            try:
                d = torch.load(f, weights_only=False)
                if hasattr(d, 'x') and hasattr(d, 'y') and hasattr(d, 'edge_index'):
                    if best is None or d.x.shape[0] > best.x.shape[0]:
                        best = d
            except Exception:
                continue
        if best is None:
            self._blocked("local_processed: 无有效 Data 文件")
        self.global_data = best
        self.global_data.num_classes = int(best.y.max().item()) + 1
        self.global_dataset = type('LD', (), {
            'data': best, 'num_classes': self.global_data.num_classes,
            'num_features': int(best.x.shape[1])})()



    def load_data(self):
        print("loading graph...")
        self.load_global_graph()
        self.feat_dim = self.global_dataset.num_features
        self.out_dim = self.global_dataset.num_classes
        self.global_data = self.global_dataset.data
        import json as _json

        if getattr(self, 'partition_mode', 'natural') == 'frozen':
            # frozen: 只加载原始图, 子图由 frozen split artifact 重建
            self.subgraphs = []
            self.node_dict = {}
            self.partition_moves = []
            return

        self.subgraphs = [self.get(i) for i in range(self.num_clients)]
        for i in range(len(self.subgraphs)):
            self.subgraphs[i].feat_dim = self.global_dataset.num_features
            self.subgraphs[i].out_dim = self.out_dim
        if self.name in ["ogbn-arxiv", "ogbn-products"]:
            for i in range(self.num_clients):
                self.subgraphs[i].y = self.subgraphs[i].y.squeeze()
        # 受控支持度划分的社区移动记录
        moves_path = osp.join(self.processed_dir, 'partition_moves.json')
        if osp.exists(moves_path):
            with open(moves_path) as f:
                self.partition_moves = _json.load(f)
        else:
            self.partition_moves = []
        # 节点分配 (全局节点 ID -> 客户端)
        assign_path = osp.join(self.processed_dir, 'node_assignments.json')
        if osp.exists(assign_path):
            with open(assign_path) as f:
                raw = _json.load(f)
            self.node_dict = {int(k): [int(v) for v in vals]
                              for k, vals in raw.items()}
        else:
            self.node_dict = {ci: list(range(self.subgraphs[ci].x.shape[0]))
                              for ci in range(self.num_clients)}

        