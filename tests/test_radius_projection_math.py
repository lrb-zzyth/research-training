"""Radius projection math/regression tests (CPU synthetic)."""
import os
import sys
import types
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_fedtad import project_to_pretrain_radius, make_server_rng
from model import ConditionalDiffusionGenerator


def test_projection_scale_invariant_and_radial_gradient_zero():
    torch.manual_seed(0)
    z = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    bank = torch.rand(3, 4, dtype=torch.float64) + 0.5
    p1 = project_to_pretrain_radius(z, labels, bank)
    p10 = project_to_pretrain_radius(10.0 * z, labels, bank)
    p1m = project_to_pretrain_radius(1e6 * z, labels, bank)
    torch.testing.assert_close(p1, p10, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(p1, p1m, rtol=1e-10, atol=1e-10)
    w = torch.randn_like(p1)
    g, = torch.autograd.grad((p1 * w).sum(), z)
    radial = (g * z).sum(dim=1)
    assert radial.abs().max().item() < 1e-9


def test_feature_norm_is_constant_under_radius_projection():
    torch.manual_seed(1)
    z = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    bank = torch.rand(3, 4, dtype=torch.float64) + 0.5
    p = project_to_pretrain_radius(z, labels, bank)
    g, = torch.autograd.grad((p ** 2).mean(), z)
    assert g.norm().item() < 1e-10


def test_dedicated_server_rng_checkpointed_matches_full():
    torch.manual_seed(2)
    base = ConditionalDiffusionGenerator(
        8, 3, 16, 6, 1e-4, 0.5, output_bound='none',
        skip_mode='timestep_scalar', use_posterior_variance=True)
    state = base.state_dict()
    labels = torch.tensor([0, 1, 2, 0])
    init = torch.randn(4, 8, generator=make_server_rng('cpu', 2024, 0, 100))

    def run(mode, segments=1):
        g = ConditionalDiffusionGenerator(
            8, 3, 16, 6, 1e-4, 0.5, output_bound='none',
            skip_mode='timestep_scalar', use_posterior_variance=True)
        g.load_state_dict(state)
        rev = make_server_rng('cpu', 2024, 0, 101)
        out = g.differentiable_sample(
            labels, backprop_mode=mode, checkpoint_segments=segments,
            apply_output_bound=False, initial_noise=init,
            noise_generator=rev)
        loss = out.square().mean()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in g.named_parameters()
                 if p.grad is not None}
        return out.detach(), grads

    ref_out, ref_g = run('full')
    for seg in (1, 2, 3, 6):
        out, grads = run('checkpointed', seg)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        for k in ref_g:
            torch.testing.assert_close(grads[k], ref_g[k], rtol=0, atol=0)
