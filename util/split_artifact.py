"""
冻结数据划分 artifact (阶段三)

将 dataset + 节点分配 + train/val/test 索引 + 标签映射 + support 配置
固化为磁盘文件, static/dynamic 等严格配对实验直接加载,
不再重新运行 Louvain 或重新划分。

内容:
  split_config.json        所有划分相关参数 (partition/allocation/split seed 等)
  client_assignments.json  每客户端的全局节点 ID
  train_val_test_indices.json  每客户端本地 train/val/test 索引
  label_mapping.json       标签映射 (anomaly_binary)
  data_identity.json       data_identity_hash (严格配对依据)
  class_support_matrix.csv 每客户端每类 train/val/test support
"""
import csv
import hashlib
import json
import os

import torch


def data_identity_hash(dataset_name, task_mode, num_classes, subgraphs,
                       mapping_info, node_dict, support_mode,
                       support_params):
    """数据身份哈希: 覆盖节点分配、划分索引、标签映射、support 配置。"""
    parts = [f"dataset={dataset_name}", f"task_mode={task_mode}",
             f"num_classes={num_classes}", f"support_mode={support_mode}",
             f"support_params={json.dumps(support_params, sort_keys=True)}"]
    if mapping_info is not None:
        parts.append(f"mapping={json.dumps(mapping_info.get('mapping', {}), sort_keys=True)}")
    # 每客户端全局节点列表 (排序保证确定性)
    for ci in range(len(subgraphs)):
        nodes = sorted(node_dict.get(ci, []))
        parts.append(f"c{ci}_nodes={json.dumps(nodes)}")
    # 每客户端 train/val/test 索引 (本地索引排序)
    for ci, sg in enumerate(subgraphs):
        for split_name in ('train_idx', 'val_idx', 'test_idx'):
            mask = sg[split_name]
            idx = sorted(int(i) for i in torch.where(mask)[0].tolist())
            parts.append(f"c{ci}_{split_name}={json.dumps(idx)}")
    payload = '|'.join(parts)
    if os.environ.get('FEDTAD_HASH_DEBUG'):
        print('HASHDEBUG payload:', payload[:1200])
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def save_split_artifact(out_dir, args, dataset, subgraphs, num_classes,
                        mapping_info, node_dict, support_params):
    """保存 frozen split artifact。"""
    os.makedirs(out_dir, exist_ok=True)
    dhash = data_identity_hash(
        dataset.name, args.task_mode, num_classes, subgraphs,
        mapping_info, node_dict, args.split_support_mode, support_params)

    split_config = {
        'dataset': dataset.name,
        'task_mode': args.task_mode,
        'num_clients': args.num_clients,
        'partition': args.partition,
        'partition_seed': int(args.partition_seed),
        'allocation_seed': int(args.allocation_seed),
        'split_seed': int(args.split_seed),
        'split_support_mode': args.split_support_mode,
        'support_params': support_params,
        'train_val_test_ratios': {'train': 0.2, 'val': 0.4, 'test': 0.4},
        'data_identity_hash': dhash,
    }
    with open(os.path.join(out_dir, 'split_config.json'), 'w') as f:
        json.dump(split_config, f, indent=1)

    with open(os.path.join(out_dir, 'client_assignments.json'), 'w') as f:
        json.dump({str(k): v for k, v in node_dict.items()}, f)

    indices = {}
    for ci, sg in enumerate(subgraphs):
        indices[str(ci)] = {
            split_name: [int(i) for i in torch.where(sg[split_name])[0].tolist()]
            for split_name in ('train_idx', 'val_idx', 'test_idx')}
    with open(os.path.join(out_dir, 'train_val_test_indices.json'), 'w') as f:
        json.dump(indices, f, indent=1)

    if mapping_info is not None:
        with open(os.path.join(out_dir, 'label_mapping.json'), 'w') as f:
            json.dump({'normal_classes': mapping_info.get('normal_classes'),
                       'anomaly_classes': mapping_info.get('anomaly_classes'),
                       'mapping': mapping_info.get('mapping')}, f, indent=1)

    with open(os.path.join(out_dir, 'data_identity.json'), 'w') as f:
        json.dump({'data_identity_hash': dhash}, f, indent=1)

    # class support matrix
    with open(os.path.join(out_dir, 'class_support_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['client', 'split', 'class', 'support'])
        for ci, sg in enumerate(subgraphs):
            for split_name in ('train_idx', 'val_idx', 'test_idx'):
                mask = sg[split_name]
                for c in range(num_classes):
                    w.writerow([ci, split_name, c,
                                int((sg.y[mask] == c).sum())])
    return dhash


def load_artifact(artifact_dir):
    """加载 frozen split artifact。"""
    with open(os.path.join(artifact_dir, 'split_config.json')) as f:
        split_config = json.load(f)
    with open(os.path.join(artifact_dir, 'client_assignments.json')) as f:
        raw = json.load(f)
    node_dict = {int(k): [int(v) for v in vals] for k, vals in raw.items()}
    with open(os.path.join(artifact_dir, 'train_val_test_indices.json')) as f:
        raw_idx = json.load(f)
    indices = {int(k): {sn: [int(i) for i in v]
                        for sn, v in vals.items()}
               for k, vals in raw_idx.items()}
    mapping = None
    mp_path = os.path.join(artifact_dir, 'label_mapping.json')
    if os.path.exists(mp_path):
        with open(mp_path) as f:
            mapping = json.load(f)
    with open(os.path.join(artifact_dir, 'data_identity.json')) as f:
        dhash = json.load(f)['data_identity_hash']
    return {'split_config': split_config, 'node_dict': node_dict,
            'indices': indices, 'mapping': mapping,
            'data_identity_hash': dhash}


def build_subgraphs_from_artifact(dataset, artifact, num_classes, device='cpu'):
    """
    从 artifact 重建子图 (不运行 Louvain/不重新划分)。
    边构造与 construct_subgraph_dict_from_node_dict 一致。
    """
    G = dataset.global_data
    node_dict = artifact['node_dict']
    indices = artifact['indices']
    subgraphs = []
    for ci in range(len(node_dict)):
        global_nodes = node_dict[ci]
        local_map = {g: l for l, g in enumerate(global_nodes)}
        x = G.x[global_nodes]
        y = G.y[global_nodes]
        edge_idx = []
        # 子图内部边 (双向)
        node_set = set(global_nodes)
        rows, cols = G.edge_index
        for i in range(G.edge_index.shape[1]):
            u, v = int(rows[i]), int(cols[i])
            if u in node_set and v in node_set:
                edge_idx.append((local_map[u], local_map[v]))
        if edge_idx:
            ei = torch.tensor(edge_idx, dtype=torch.long).T
        else:
            ei = torch.zeros((2, 1), dtype=torch.long)
        from torch_geometric.data import Data
        sg = Data(x=x, y=y, edge_index=ei)
        for split_name in ('train_idx', 'val_idx', 'test_idx'):
            idx_list = indices[ci][split_name]
            mask = torch.zeros(len(global_nodes), dtype=torch.bool)
            mask[idx_list] = True
            sg[split_name] = mask
        sg.feat_dim = int(x.shape[1])
        sg.out_dim = num_classes
        subgraphs.append(sg.to(device))
    return subgraphs


def initialization_hash(seed, feat_dim, num_classes, hid_dim, dropout, num_clients):
    """初始化身份哈希: 由模型初始化 RNG 状态决定 (同 model_seed -> 同初始化)。"""
    # 在模型构造后的固定点捕获 RNG 状态
    state = torch.get_rng_state().clone()
    return hashlib.sha1(state.numpy().tobytes()).hexdigest()[:16]
