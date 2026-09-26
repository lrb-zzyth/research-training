"""Final v3 math/numerical regression tests.

Designed to run on CPU even when torch_geometric is not installed: PyG symbols are
stubbed only for importing modules whose tested functions do not depend on PyG.
"""
import importlib
import os
import re
import sys
import types

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from util.kl_utils import categorical_kl_from_logits
from util.projection_utils import project_to_pretrain_radius


def _install_pyg_stubs():
    if 'torch_geometric' in sys.modules:
        return
    tg = types.ModuleType('torch_geometric')
    tg_nn = types.ModuleType('torch_geometric.nn')
    tg_utils = types.ModuleType('torch_geometric.utils')
    tg_data = types.ModuleType('torch_geometric.data')
    tg_convert = types.ModuleType('torch_geometric.utils.convert')

    class DummyGCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.lin(x)

    class DummyData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

    tg_nn.GCNConv = DummyGCNConv
    tg_data.Data = DummyData
    tg_data.Dataset = DummyDataset
    tg_utils.to_dense_adj = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('stub to_dense_adj should not be called in this test'))
    tg_utils.add_self_loops = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('stub add_self_loops should not be called in this test'))
    tg_utils.dense_to_sparse = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('stub dense_to_sparse should not be called in this test'))
    tg_convert.to_networkx = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('stub to_networkx should not be called in this test'))
    tg_utils.convert = tg_convert

    tg.nn = tg_nn
    tg.utils = tg_utils
    tg.data = tg_data
    sys.modules['torch_geometric'] = tg
    sys.modules['torch_geometric.nn'] = tg_nn
    sys.modules['torch_geometric.utils'] = tg_utils
    sys.modules['torch_geometric.data'] = tg_data
    sys.modules['torch_geometric.utils.convert'] = tg_convert


def test_kl_direction_is_global_to_local():
    g = torch.tensor([[3.0, 0.5, -1.0], [0.2, -0.7, 1.1]], requires_grad=True)
    l = torch.tensor([[0.1, 1.5, -0.2], [1.2, -0.4, 0.0]], requires_grad=True)
    got = categorical_kl_from_logits(g, l)
    p = torch.distributions.Categorical(logits=g)
    q = torch.distributions.Categorical(logits=l)
    ref = torch.distributions.kl.kl_divergence(p, q).mean()
    rev = torch.distributions.kl.kl_divergence(q, p).mean()
    torch.testing.assert_close(got, ref)
    assert abs(float(got.detach() - rev.detach())) > 1e-3, 'test logits must distinguish KL direction'
    gg, gl = torch.autograd.grad(got, (g, l))
    assert torch.isfinite(gg).all() and torch.isfinite(gl).all()
    assert gg.norm() > 0 and gl.norm() > 0


def test_stable_projection_handles_huge_finite_raw_values():
    raw = torch.tensor([[1e30, -2e30, 3e30], [-4e30, 1e30, 2e30]],
                       dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 1])
    bank = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
    out = project_to_pretrain_radius(raw, labels, bank)
    assert torch.isfinite(out).all()
    norms = torch.linalg.vector_norm(out.double(), dim=1)
    torch.testing.assert_close(norms, torch.tensor([2.0, 3.0], dtype=torch.float64),
                               rtol=1e-6, atol=1e-6)
    w = torch.tensor([[0.4, -0.1, 0.7], [0.3, 0.2, -0.5]], dtype=torch.float32)
    grad, = torch.autograd.grad((out * w).sum(), raw)
    radial = (grad.double() * raw.detach().double()).sum(dim=1)
    denom = grad.double().norm(dim=1) * raw.detach().double().norm(dim=1)
    ratio = radial.abs() / denom.clamp_min(1e-30)
    assert ratio.max().item() < 1e-5



def test_infonce_denominator_includes_positive():
    _install_pyg_stubs()
    tu = importlib.import_module('util.task_util')
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z2 = torch.tensor([[0.9, 0.1], [0.1, 0.9], [1.0, 0.8]])
    tau = 0.2
    got = tu.subgraph_contrastive_loss(z1, z2, tau=tau)
    a = torch.nn.functional.normalize(z1, dim=1)
    b = torch.nn.functional.normalize(z2, dim=1)
    sim = a @ b.T / tau
    ref = -torch.log_softmax(sim, dim=1).diag().mean()
    torch.testing.assert_close(got, ref)


def test_edge_perturbation_preserves_undirected_pairing():
    _install_pyg_stubs()
    tu = importlib.import_module('util.task_util')
    # Complete-ish small undirected graph stored in both directions.
    edges = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,0),(0,3),(0,2),(2,0)]
    ei = torch.tensor(edges, dtype=torch.long).T
    import random
    random.seed(11)
    out = tu.edge_perturbation(ei, 4, drop_rate=0.4)
    pairs = {(int(u), int(v)) for u, v in out.T.tolist()}
    assert all(u != v for u, v in pairs)
    assert all((v, u) in pairs for u, v in pairs)

def test_rwr_never_injects_other_connected_components():
    _install_pyg_stubs()
    tu = importlib.import_module('util.task_util')
    # Two disconnected components: 0--1 and 2--3--4.  Request more nodes than
    # anchor component contains; result must stay {0,1} instead of global fill.
    edge_index = torch.tensor([[0, 1, 2, 3, 3, 4],
                               [1, 0, 3, 2, 4, 3]], dtype=torch.long)
    subgraphs, centers = tu.rwr_subgraph_sampling(
        edge_index, 5, [0], subgraph_size=4, restart_prob=0.5,
        max_length=2, seed=123)
    assert set(subgraphs[0]).issubset({0, 1})
    assert subgraphs[0][centers[0]] == 0


def test_rwr_local_fallback_can_fill_reachable_nodes():
    _install_pyg_stubs()
    tu = importlib.import_module('util.task_util')
    # max_length=0 forces fallback; BFS must still fill from anchor component.
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    subgraphs, centers = tu.rwr_subgraph_sampling(
        edge_index, 4, [0], subgraph_size=2, restart_prob=1.0,
        max_length=0, seed=7)
    sg = subgraphs[0]
    assert len(sg) == 3
    assert set(sg).issubset({0, 1, 2, 3})
    assert sg[centers[0]] == 0


def test_ddpm_checkpointed_matches_full_exactly():
    _install_pyg_stubs()
    model_mod = importlib.import_module('model')
    Gen = model_mod.ConditionalDiffusionGenerator
    torch.manual_seed(22)
    base = Gen(6, 3, hidden_dim=12, num_steps=6, beta_start=1e-4,
               beta_end=0.5, output_bound='none', skip_mode='timestep_scalar')
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    labels = torch.tensor([0, 1, 2, 0])
    init_gen = torch.Generator().manual_seed(100)
    init = torch.randn(4, 6, generator=init_gen)

    def run(mode, segments):
        g = Gen(6, 3, hidden_dim=12, num_steps=6, beta_start=1e-4,
                beta_end=0.5, output_bound='none', skip_mode='timestep_scalar')
        g.load_state_dict(state)
        rev = torch.Generator().manual_seed(101)
        out = g.differentiable_sample(
            labels, backprop_mode=mode, checkpoint_segments=segments,
            apply_output_bound=False, initial_noise=init,
            noise_generator=rev)
        loss = out.square().mean()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in g.named_parameters()
                 if p.grad is not None}
        return out.detach(), grads

    ref_out, ref_grads = run('full', 1)
    for seg in (1, 2, 3, 6):
        out, grads = run('checkpointed', seg)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        assert grads.keys() == ref_grads.keys()
        for k in grads:
            torch.testing.assert_close(grads[k], ref_grads[k], rtol=0, atol=0)



def test_train_generator_and_student_kl_paths_are_global_to_local():
    _install_pyg_stubs()
    tf = importlib.import_module('train_fedtad')
    Data = sys.modules['torch_geometric.data'].Data

    class TinyModel(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight.clone())
        def forward(self, data):
            return data.x @ self.weight

    wg = torch.tensor([[1.0, -0.4], [0.3, 0.8]], dtype=torch.float32)
    wt = torch.tensor([[-0.2, 0.7], [1.1, -0.5]], dtype=torch.float32)
    global_model = TinyModel(wg)
    teacher = TinyModel(wt)
    for p in global_model.parameters():
        p.requires_grad_(False)
    for p in teacher.parameters():
        p.requires_grad_(False)

    x = torch.tensor([[0.7, -0.2], [0.1, 1.3]],
                     dtype=torch.float32, requires_grad=True)
    graph = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
    labels = torch.tensor([0, 1])
    ckr = torch.ones(1, 2)

    got = tf.compute_generator_disagreement_loss(
        graph, labels, [teacher], global_model, ckr, 2, 'cpu', loss_type='kl')
    gl = global_model(graph)
    tl = teacher(graph)
    ref = (categorical_kl_from_logits(gl[:1], tl[:1])
           + categorical_kl_from_logits(gl[1:], tl[1:]))
    torch.testing.assert_close(got, ref)
    gx, = torch.autograd.grad(got, x)
    assert torch.isfinite(gx).all() and gx.norm() > 0

    for p in global_model.parameters():
        p.requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    graph2 = Data(x=x2, edge_index=graph.edge_index)
    got_student = tf.compute_student_distillation_loss(
        graph2, labels, [teacher], global_model, ckr, 2, 'cpu',
        temperature=1.0, loss_type='kl')
    gl2 = global_model(Data(x=x2.detach(), edge_index=graph.edge_index))
    tl2 = teacher(Data(x=x2.detach(), edge_index=graph.edge_index))
    ref_student = (categorical_kl_from_logits(gl2[:1], tl2[:1])
                   + categorical_kl_from_logits(gl2[1:], tl2[1:]))
    torch.testing.assert_close(got_student, ref_student)
    got_student.backward()
    assert global_model.weight.grad is not None
    assert torch.isfinite(global_model.weight.grad).all()
    assert x2.grad is None, 'student distillation must detach fake graph features'

def test_train_wiring_uses_explicit_kl_and_checkpoint_segments():
    src = open(os.path.join(REPO, 'train_fedtad.py'), encoding='utf-8').read()
    # Both generator disagreement and student distillation must pass global logits
    # as P and local/teacher logits as Q to the explicit KL helper.
    assert src.count('categorical_kl_from_logits(') >= 2
    assert re.search(r'categorical_kl_from_logits\(\s*global_logits\[idx_c\],\s*t_logits\[idx_c\]', src)
    assert 'checkpoint_segments=args.checkpoint_segments' in src


def test_formal_campaign_bumped_to_v3():
    src = open(os.path.join(REPO, 'experiments', 'accuracy_benchmark',
                            'formal_campaign.py'), encoding='utf-8').read()
    assert "FORMAL_VERSION = 'v3'" in src
    assert "'kl_direction': 'KL(global||local)" in src


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn()
        print(fn.__name__, 'PASS')
    print(f'ALL {len(tests)} FINAL V3 TESTS PASSED')
