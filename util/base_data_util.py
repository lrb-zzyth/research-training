import time
import numpy as np
import scipy.sparse as sp
import random
import torch
from torch import Tensor
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data

def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def to_undirected(edge_index):
    if isinstance(edge_index, sp.csr_matrix) or isinstance(edge_index, sp.coo_matrix):
        row, col = edge_index.row, edge_index.col
        row, col = torch.from_numpy(row), torch.from_numpy(col)
    else:
        row, col = edge_index
        if not isinstance(row, Tensor) or not isinstance(col, Tensor):
            row, col = torch.from_numpy(row), torch.from_numpy(col)
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)
    return new_edge_index


def idx_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def louvain_partition(graph, num_clients, delta=20, return_groups=False,
                      louvain_seed=2024):
    # Lazy import: cached/frozen partitions do not need the vendored Louvain module.
    # Rebuilding a partition still fails fast unless the original louvain/ tree exists.
    try:
        from louvain.community import community_louvain
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing vendored louvain/ module; cannot rebuild the Louvain partition. "
            "Copy louvain/ from the complete project; do not silently substitute a "
            "different implementation because that can change client partitions.") from e
    num_nodes = graph.number_of_nodes()

    # 固定 Louvain 随机种子: 保证划分可复现 (python-louvain 默认不固定 RNG)
    # 注意: 磁盘缓存(natural)保持旧划分不变; 新缓存目录(如 enriched)使用固定种子
    partition = community_louvain.best_partition(graph, random_state=louvain_seed)

    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max = num_nodes // num_clients - delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {
        k: v
        for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)
    }

    owner_node_ids = {owner_id: [] for owner_id in range(num_clients)}

    owner_nodes_len = num_nodes // num_clients
    owner_list = [i for i in range(num_clients)]
    owner_ind = 0

    bad_key = 1000

    for group_i in sort_len_dict.keys():
        while (
            len(owner_list) >= 2
            and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len
        ):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        cnt = 0
        while (
            len(owner_node_ids[owner_list[owner_ind]]) +
                len(partition_groups[group_i])
            >= owner_nodes_len + delta
        ):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > bad_key:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[owner_ind]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[owner_ind]])
                        owner_ind = i
                break

        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    node_dict = owner_node_ids

    print("end louvain")
    if return_groups:
        groups = [list(partition_groups[g]) for g in groups]
        return node_dict, groups
    return node_dict


def enrich_anomaly_support(node_dict, groups, y, anomaly_classes, target,
                           num_clients, delta=20, seed=0):
    """
    受控支持度划分 (anomaly_enriched_partition, 仅机制实验):
    在不复制节点、不修改标签、不产生跨客户端重复的前提下,
    通过移动整个 Louvain 社区使更多客户端达到最小异常节点数 target。

    规则:
      - 只移动完整社区 (不拆分)
      - 移动后: 接收方大小 <= floor(N/K) + 2*delta, 来源方大小 >= floor(N/K) - 2*delta
      - 确定性: 客户端/社区顺序由 seed 控制的 RNG 决定
      - 记录每次移动

    Returns:
        (new_node_dict, moves)  moves: list of dict
    """
    import random as _random
    rng = _random.Random(int(seed))
    N = sum(len(nodes) for nodes in node_dict.values())
    floor_size = N // num_clients

    y_arr = y.cpu().numpy() if hasattr(y, 'cpu') else y
    anom_set = set(anomaly_classes)
    group_owner = {}
    group_anom = {}
    for gi, nodes in enumerate(groups):
        group_anom[gi] = sum(1 for n in nodes if int(y_arr[n]) in anom_set)

    def client_anom(ci):
        return sum(1 for n in node_dict[ci] if int(y_arr[n]) in anom_set)

    def owner_of(gi):
        for cj, nodes_c in node_dict.items():
            if any(n in nodes_c for n in groups[gi][:1]):
                return cj
        return None

    def move_group(gi, to_ci, from_r):
        for n in groups[gi]:
            node_dict[to_ci].append(n)
            node_dict[from_r].remove(n)
        moved_groups.add(gi)

    moves = []
    moved_groups = set()
    size_hi = floor_size + 4 * delta  # 接收方容量上限
    size_lo = floor_size - 4 * delta  # 来源方容量下限
    for iter_ in range(300):
        deficit = [(ci, client_anom(ci)) for ci in range(num_clients)
                   if client_anom(ci) < target]
        if not deficit:
            break
        deficit.sort(key=lambda x: x[1])
        ci = deficit[0][0]  # 确定性: 始终服务最不足的客户端
        room = size_hi - len(node_dict[ci])

        # 1) 直接移动: 小组放进接收方剩余容量
        candidates = []
        for gi, nodes in enumerate(groups):
            if gi in moved_groups or group_anom[gi] <= 0:
                continue
            r = owner_of(gi)
            if r is None or r == ci:
                continue
            if client_anom(r) - group_anom[gi] < target:
                continue  # 来源方移后跌破 target
            sz_g = len(nodes)
            if sz_g <= room and len(node_dict[r]) - sz_g >= size_lo:
                candidates.append((group_anom[gi], sz_g, gi, r))
        if candidates:
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: (-x[0], x[1]))
            _, sz_g, gi, r = candidates[0]
            move_group(gi, ci, r)
            moves.append({'type': 'move', 'group_id': gi, 'from_client': r,
                          'to_client': ci, 'group_size': sz_g,
                          'anomalies': group_anom[gi]})
            continue

        # 2) 交换: 大异常组与接收方的大非异常组互换 (尺寸守恒, 集中异常)
        swaps = []
        for gi, nodes in enumerate(groups):
            if gi in moved_groups or group_anom[gi] <= 0:
                continue
            r = owner_of(gi)
            if r is None or r == ci:
                continue
            if client_anom(r) - group_anom[gi] < target:
                continue
            sz_a = len(nodes)
            for gj, nodes_b in enumerate(groups):
                if gj in moved_groups or gj == gi:
                    continue
                if owner_of(gj) != ci:
                    continue
                sz_b = len(nodes_b)
                gain = group_anom[gi] - group_anom[gj]
                if gain <= 0:
                    continue
                if abs(sz_a - sz_b) > 2 * delta:  # 尺寸接近才交换
                    continue
                if (len(node_dict[r]) - sz_a + sz_b >= size_lo
                        and len(node_dict[r]) - sz_a + sz_b <= size_hi
                        and len(node_dict[ci]) - sz_b + sz_a <= size_hi):
                    swaps.append((gain, sz_a, gi, gj, r))
        if swaps:
            rng.shuffle(swaps)
            swaps.sort(key=lambda x: (-x[0], x[1]))
            _, sz_a, gi, gj, r = swaps[0]
            move_group(gi, ci, r)
            move_group(gj, r, ci)
            moves.append({'type': 'swap', 'group_id': gi, 'pair_group_id': gj,
                          'from_client': r, 'to_client': ci,
                          'group_size': sz_a, 'pair_group_size': len(groups[gj]),
                          'anomalies': group_anom[gi],
                          'anomalies_received_back': group_anom[gj]})
            continue
        break
    return node_dict, moves


def data_partition(G, num_clients, train, val, test, partition, part_delta,
                   support_mode='natural', anomaly_classes=None,
                   anomaly_partition_target=0, support_rebalance_seed=0,
                   min_train_support=0, partition_seed=None,
                   return_node_dict=False):
    """
    support_mode:
      natural                : 原始 Louvain 划分 (默认)
      anomaly_enriched       : 受控支持度划分 (仅机制实验, 移动完整社区)
    partition_seed: Louvain 随机种子 (None=用固定默认 2024)
    """
    time_st = time.time()
    print("start g to nxg")
    graph_nx = to_networkx(G, to_undirected=True, remove_self_loops=True)
    print(f"get nxg {time.time()-time_st} sec.")

    partition_moves = []
    if partition == "Louvain":
        print("Conducting louvain graph partition...")
        louvain_seed = partition_seed if partition_seed is not None else 2024
        if support_mode == 'anomaly_enriched':
            assert anomaly_classes, "anomaly_enriched 需要 anomaly_classes"
            assert anomaly_partition_target > 0, "anomaly_enriched 需要 anomaly_partition_target > 0"
            node_dict, groups = louvain_partition(
                graph=graph_nx, num_clients=num_clients, delta=part_delta,
                return_groups=True, louvain_seed=louvain_seed)
            node_dict, partition_moves = enrich_anomaly_support(
                node_dict, groups, G.y, anomaly_classes,
                target=anomaly_partition_target,
                num_clients=num_clients, delta=part_delta,
                seed=support_rebalance_seed)
            # 持久化社区移动记录 (FGLDataset.process 写入 moves.json)
            G.partition_moves = partition_moves
        else:
            node_dict = louvain_partition(
                graph=graph_nx, num_clients=num_clients, delta=part_delta,
                louvain_seed=louvain_seed)
    elif partition == "Metis":
        # import metispy as metis
        import metis

        print("Conducting metis graph partition...")
        node_dict = {}
        n_cuts, membership = metis.part_graph(graph_nx, num_clients)
        for client_id in range(num_clients):
            client_indices = np.where(np.array(membership) == client_id)[0]
            client_indices = list(client_indices)
            node_dict[client_id] = client_indices
    else:
        raise ValueError(f"No such partition method: '{partition}'.")

    assert sum([len(node_dict[i])
               for i in range(len(node_dict))]) == G.num_nodes
    

            
    subgraph_list = construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=num_clients,
        node_dict=node_dict,
        graph_nx=graph_nx,
        train=train,
        val=val,
        test=test,
    )
    
    G.num_samples = 0
    for subgraph in subgraph_list:
        G.num_samples += subgraph.num_samples
    if return_node_dict:
        return subgraph_list, node_dict
    return subgraph_list


def analysis_graph_structure_homo_hete_info(G):
    structure_homo_hete_label_info = {}
    structure_homo_hete_label_info["node_homophily"] = label_node_homogeneity(
        G)
    structure_homo_hete_label_info["edge_homophily"] = label_edge_homogeneity(
        G)
    print(
        f"homo_node: {structure_homo_hete_label_info['node_homophily']:.4f}\nhomo_edge: {structure_homo_hete_label_info['edge_homophily']:.4f}"
    )
    return (
        structure_homo_hete_label_info["node_homophily"],
        structure_homo_hete_label_info["edge_homophily"],
    )


def label_node_homogeneity(G):
    num_nodes = G.num_nodes
    homophily = 0
    for edge_u in range(num_nodes):
        hit = 0
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                if G.y[edge_u] == G.y[edge_v]:
                    hit += 1
            homophily += hit / len(edge_v_list)
    homophily /= num_nodes
    return homophily


def label_edge_homogeneity(G):
    num_edges = G.num_edges
    homophily = 0
    for i in range(num_edges):
        if G.y[G.edge_index[0][i]] == G.y[G.edge_index[1][i]]:
            homophily += 1
    homophily /= num_edges
    return homophily



def construct_subgraph_dict_from_node_dict(num_clients, node_dict, G, graph_nx, train, val, test):
    subgraph_list = []
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        print(f"num_local_nodes for c {client_id}: {num_local_nodes}")
        
        train_idx = []
        val_idx = []
        test_idx = []
        class_i_idx_list = {}
        
        for idx in range(num_local_nodes):
            label = G.y[node_dict[client_id][idx]]
            if int(label) not in class_i_idx_list:
                class_i_idx_list[int(label)] = []
            class_i_idx_list[int(label)].append(idx)
        
        for class_i in range(G.num_classes):
            if class_i not in class_i_idx_list:
                continue
            local_node_idx = class_i_idx_list[class_i]
            random.shuffle(local_node_idx)

            num_local_nodes_class_i = len(local_node_idx)
            train_size = int(num_local_nodes_class_i * train)
            val_size = int(num_local_nodes_class_i * val)
            test_size = int(num_local_nodes_class_i * test) 

            train_idx += local_node_idx[:train_size]
            val_idx += local_node_idx[train_size: train_size + val_size]
            test_idx += local_node_idx[train_size + val_size:]

        assert len(train_idx) + len(val_idx) + len(test_idx) == num_local_nodes

        local_train_idx = idx_to_mask(train_idx, size=num_local_nodes)
        local_val_idx = idx_to_mask(val_idx, size=num_local_nodes)
        local_test_idx = idx_to_mask(test_idx, size=num_local_nodes)

        node_idx_map = {}
        edge_idx = []
        for idx in range(num_local_nodes):
            node_idx_map[node_dict[client_id][idx]] = idx
        edge_idx += [
            (node_idx_map[x[0]], node_idx_map[x[1]])
            for x in graph_nx.subgraph(node_dict[client_id]).edges
        ]
        edge_idx += [
            (node_idx_map[x[1]], node_idx_map[x[0]])
            for x in graph_nx.subgraph(node_dict[client_id]).edges
        ]

        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T
        subgraph = Data(
            x=G.x[node_dict[client_id]],
            y=G.y[node_dict[client_id]],
            edge_index=edge_idx_tensor,
        )


        subgraph.train_idx = local_train_idx
        subgraph.val_idx = local_val_idx
        subgraph.test_idx = local_test_idx
        subgraph.num_samples = subgraph.num_nodes
        subgraph_list.append(subgraph)
        print(
            "Client: {}\tTotal Nodes: {}\tTotal Edges: {}\tTrain Nodes: {}\tVal Nodes: {}\tTest Nodes\t{}".format(
                client_id + 1,
                subgraph.num_nodes,
                subgraph.num_edges,
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )
        )

    return subgraph_list
