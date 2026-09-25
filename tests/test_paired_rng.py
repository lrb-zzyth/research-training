"""
Paired-arm RNG 解耦回归测试:

  同一 (round, client) 的本地训练随机流必须与服务器侧 RNG 消耗完全无关。
  Arm C: 直接 reseed + 训练
  Arm B: 先执行大量 server-only 随机操作 (torch + python random), 再 reseed + 训练
  => 参数必须逐位一致 (覆盖 dropout 与 python random 增强路径)。

CPU only, 合成配置。
"""
import os
import random
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_fedtad import reseed_client_rng
from util.task_util import edge_perturbation


class TinyDrop(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        return self.fc(self.drop(x))


def _template_state():
    torch.manual_seed(0)
    return TinyDrop().state_dict()


def _local_train(m, x, y):
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.train()
    opt.zero_grad()
    loss = F.mse_loss(m(x), y)
    loss.backward()
    opt.step()
    return [p.detach().clone() for p in m.parameters()]


def _server_only_random_ops():
    """模拟 B-only 初始化路径的全局 RNG 消耗 (Embedding init / 采样等)。"""
    torch.randn(1000, 32)
    torch.randperm(500)
    for _ in range(200):
        random.random()
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_perturbation(ei, 4, 0.2)   # 消耗 python random


def test_reseed_decouples_server_rng():
    args = types.SimpleNamespace(seed=2024)
    x = torch.randn(16, 8)
    y = torch.randn(16, 8)

    # Arm C: 直接 reseed + 训练
    reseed_client_rng(args, 0, 0)
    mC = TinyDrop()
    mC.load_state_dict(_template_state())
    pC = _local_train(mC, x, y)

    # Arm B: 先大量消耗全局 RNG, 再 reseed + 训练
    _server_only_random_ops()
    reseed_client_rng(args, 0, 0)
    mB = TinyDrop()
    mB.load_state_dict(_template_state())
    pB = _local_train(mB, x, y)

    for a, b in zip(pC, pB):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("paired-rng: server RNG 消耗不影响客户端训练轨迹 (逐位一致, 含 dropout): PASS")


def test_edge_perturbation_deterministic_under_reseed():
    """同一 reseed 下 edge perturbation (python random) 结果逐位一致。"""
    args = types.SimpleNamespace(seed=2024)
    ei = torch.randint(0, 30, (2, 200))
    reseed_client_rng(args, 1, 3)
    a = edge_perturbation(ei, 30, 0.2)
    _server_only_random_ops()          # 中间消耗全局 RNG
    reseed_client_rng(args, 1, 3)
    b = edge_perturbation(ei, 30, 0.2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("paired-rng: edge perturbation 确定性: PASS")


if __name__ == '__main__':
    test_reseed_decouples_server_rng()
    test_edge_perturbation_deterministic_under_reseed()
    print("\nALL PAIRED-RNG TESTS PASSED")
