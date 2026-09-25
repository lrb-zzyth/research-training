"""
Correctness patch 单元测试: 真实 FedAvg 路径下的 checkpoint/optimizer 状态管理
(Bug 5/6, 重写为独立对象 + 真实 broadcast/local-train/FedAvg 流程)。

CPU only, 合成模型, 不加载真实数据集。
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.checkpoint import (save_checkpoint, load_checkpoint,
                             find_best_checkpoint, find_final_checkpoint)


class TinyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 4)
        self.lin2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.lin2(torch.relu(self.lin1(x)))


def _build_models():
    """global 与 local 必须是独立对象, 初始参数一致。"""
    g = TinyGCN()
    l = TinyGCN()
    l.load_state_dict(g.state_dict())
    assert g is not l
    return g, l


def _local_train(model, opt, x, y):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()


def _fedavg_to_global(global_model, locals_, weights=None):
    with torch.no_grad():
        for i, m in enumerate(locals_):
            w = (weights[i] if weights else 1.0 / len(locals_))
            for gp, lp in zip(global_model.parameters(), m.parameters()):
                if i == 0:
                    gp.data.copy_(w * lp.data)
                else:
                    gp.data.add_(w * lp.data)


def _broadcast(global_model, locals_):
    for m in locals_:
        m.load_state_dict(global_model.state_dict())


def _round0(seed):
    torch.manual_seed(seed)
    g, l = _build_models()
    gopt = torch.optim.Adam(g.parameters(), lr=1e-2)
    lopt = torch.optim.Adam(l.parameters(), lr=1e-2)   # persistent, 独立对象
    assert id(gopt) != id(lopt), "global/local optimizer 必须独立"
    x = torch.randn(8, 8)
    y = torch.randn(8, 2)
    _broadcast(g, [l])
    torch.manual_seed(seed + 1)
    _local_train(l, lopt, x, y)
    _fedavg_to_global(g, [l])
    return g, l, lopt, x, y


def _round1(g, l, lopt, x, y, seed):
    _broadcast(g, [l])            # 本地 optimizer 不重建 (persistent)
    torch.manual_seed(seed + 2)
    _local_train(l, lopt, x, y)


def test_B1_persistent_continuous_vs_checkpoint_resume():
    """Arm1 连续 2 轮 vs Arm2 round0→save/resume→round1, 参数与 Adam 状态逐位一致。"""
    seed = 7
    # Arm 1: continuous
    g1, l1, lopt1, x, y = _round0(seed)
    _round1(g1, l1, lopt1, x, y, seed)

    # Arm 2: round 0 相同, 然后 checkpoint/resume
    g2, l2, lopt2, x2, y2 = _round0(seed)      # 同 seed, 同轨迹
    assert torch.equal(x, x2) and torch.equal(y, y2)
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(d, 'best.pt', g2, global_optimizer=None,
                        round_idx=0, local_optimizers=[lopt2])
        g3 = TinyGCN()
        l3 = TinyGCN()
        lopt3 = torch.optim.Adam(l3.parameters(), lr=1e-2)
        ck = load_checkpoint(os.path.join(d, 'best.pt'), global_model=g3,
                             local_optimizers=[lopt3])
        assert ck.get('local_optimizer_states') is not None
        _round1(g3, l3, lopt3, x, y, seed)

    # 参数逐位一致
    for p1, p3 in zip(g1.parameters(), g3.parameters()):
        torch.testing.assert_close(p1, p3, rtol=0, atol=0)
    for p1, p3 in zip(l1.parameters(), l3.parameters()):
        torch.testing.assert_close(p1, p3, rtol=0, atol=0)
    # Adam 状态逐位一致 (step/exp_avg/exp_avg_sq)
    s1 = lopt1.state_dict()['state']
    s3 = lopt3.state_dict()['state']
    assert set(s1) == set(s3)
    for pid in s1:
        for k in ('step', 'exp_avg', 'exp_avg_sq'):
            assert torch.equal(s1[pid][k], s3[pid][k]), \
                f"local Adam {k} 不一致"
    # identity sanity
    assert g1 is not l1 and g3 is not l3
    print("Bug5 B1 persistent continuous vs checkpoint/resume (逐位): PASS")


def test_B2_reset_each_round_fresh_adam():
    """reset_each_round: round0 训练后 state 非空; round1 广播+重建后 state 空;
    一步后 step==1。"""
    torch.manual_seed(3)
    g, l = _build_models()
    gopt = torch.optim.Adam(g.parameters(), lr=1e-2)
    lopt = torch.optim.Adam(l.parameters(), lr=1e-2)
    assert id(gopt) != id(lopt)
    x = torch.randn(8, 8)
    y = torch.randn(8, 2)
    # Round 0
    _broadcast(g, [l])
    torch.manual_seed(4)
    _local_train(l, lopt, x, y)
    assert all(len(st) > 0 for st in lopt.state_dict()['state'].values()), \
        "round0 后 local Adam state 必须非空"
    _fedavg_to_global(g, [l])
    # Round 1: broadcast + 重建 (真实 reset_each_round 语义)
    _broadcast(g, [l])
    lopt_new = torch.optim.Adam(l.parameters(), lr=1e-2)
    assert all(len(st) == 0 for st in lopt_new.state_dict()['state'].values()), \
        "重建后必须从 empty Adam state 开始"
    torch.manual_seed(5)
    _local_train(l, lopt_new, x, y)
    steps = {float(st['step']) for st in lopt_new.state_dict()['state'].values()}
    assert steps == {1.0}, f"第一步后 step 必须为 1, 得到 {steps}"
    print("Bug5 B2 reset_each_round fresh Adam: PASS")


def test_bug6_find_final_checkpoint():
    """Bug 6: best.pt > last.pt > 最新 round_N.pt > None。"""
    with tempfile.TemporaryDirectory() as d:
        assert find_final_checkpoint(d) is None, "空目录应返回 None"
        open(os.path.join(d, 'round_3.pt'), 'w').close()
        open(os.path.join(d, 'round_7.pt'), 'w').close()
        assert find_final_checkpoint(d) == os.path.join(d, 'round_7.pt')
        open(os.path.join(d, 'last.pt'), 'w').close()
        assert find_final_checkpoint(d) == os.path.join(d, 'last.pt')
        open(os.path.join(d, 'best.pt'), 'w').close()
        assert find_final_checkpoint(d) == os.path.join(d, 'best.pt')
        assert find_best_checkpoint(d) == os.path.join(d, 'best.pt')
    print("Bug6 find_final_checkpoint: PASS")


if __name__ == '__main__':
    test_B1_persistent_continuous_vs_checkpoint_resume()
    test_B2_reset_each_round_fresh_adam()
    test_bug6_find_final_checkpoint()
    print("\nALL CHECKPOINT STATE MGMT TESTS PASSED")
