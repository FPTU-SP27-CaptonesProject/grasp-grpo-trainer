import pytest
import torch

from grasp_grpo.grpo import merge_actions, token_loss


def test_on_policy_gradient_is_minus_advantage_per_token():
    old = torch.tensor([-0.5, -1.0, -2.0])
    logp = old.clone().requires_grad_(True)
    loss, stats = token_loss(logp, old, old.clone(), torch.tensor(1.5), clip_low=0.2, clip_high=0.28, kl_beta=0.002)
    loss.backward()
    assert torch.allclose(logp.grad, torch.full((3,), -1.5))
    assert stats["kl_sum"] == pytest.approx(0.0, abs=1e-7)
    assert stats["clipped"] == 0


def test_k3_kl_is_positive_when_policy_moves_away():
    old = torch.tensor([-1.0, -1.0])
    _, stats = token_loss(torch.tensor([-0.5, -1.5]), old, old, torch.tensor(0.0), clip_low=0.2, clip_high=0.28, kl_beta=0.002)
    assert stats["kl_sum"] > 0


def test_clip_higher_counts_tokens_outside_asymmetric_range():
    old = torch.zeros(3)
    logp = torch.log(torch.tensor([0.7, 1.25, 1.35]))  # ratios 0.7, 1.25, 1.35
    _, stats = token_loss(logp, old, logp, torch.tensor(1.0), clip_low=0.2, clip_high=0.28, kl_beta=0.0)
    assert stats["clipped"] == 2  # 0.7 < 0.8 and 1.35 > 1.28; 1.25 is inside
    assert stats["ratio_sum"] == pytest.approx(3.3, rel=1e-5)


def test_merge_actions_sums_advantages_of_shared_ancestors():
    shared, a, b = {"id": "rd0"}, {"id": "qf1"}, {"id": "qf2"}
    leaves = [{"advantage": 0.5, "actions": [shared, a]}, {"advantage": -0.25, "actions": [shared, b]}]
    merged = merge_actions(leaves)
    assert [(action["id"], advantage) for action, advantage in merged] == [("rd0", 0.25), ("qf1", 0.5), ("qf2", -0.25)]
