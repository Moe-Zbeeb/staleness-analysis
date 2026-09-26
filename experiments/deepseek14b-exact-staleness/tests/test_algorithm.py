import pytest
import torch

from deepseek_study.learning.advantages import group_advantages
from deepseek_study.learning.loss import clipped_grpo
from prime_rl.trainer.rl.loss import LossInputs


@pytest.mark.parametrize(
    "mode,expected", [("centered", [-0.5, 0.5]), ("population_std", [-1, 1]), ("sample_std", [-(2**-0.5), 2**-0.5])]
)
def test_group_normalization(mode, expected):
    assert group_advantages([0, 1], mode, 1e-8).tolist() == pytest.approx(expected)
    assert group_advantages([1, 1], mode, 1e-8).tolist() == [0.0, 0.0]


def test_clipping_uses_behavior_logprobs_and_masks_prompt_gradients():
    current = torch.tensor([100.0, -1.0, -3.0, -1.0], requires_grad=True)
    behavior = torch.tensor([-100.0, -2.0, -2.0, -2.0], requires_grad=True)
    inputs = LossInputs(
        current, behavior, None, torch.tensor([1.0, 1.0, -1.0, -1.0]), torch.tensor([False, True, True, True])
    )
    result = clipped_grpo(inputs, 0.2)
    assert result.loss.item() == pytest.approx(-1.2 + 0.8 + torch.e, rel=1e-6)
    result.loss.backward()
    assert current.grad.tolist() == pytest.approx([0.0, 0.0, 0.0, torch.e], rel=1e-6)
    assert behavior.grad is None
    assert result.metrics["study/ratio_outside_clip_range"].tolist() == [1.0, 1.0, 1.0]
    assert result.metrics["study/surrogate_clipped_fraction"].tolist() == [1.0, 1.0, 0.0]
    assert result.metrics["study/noncontributing_token_fraction"].tolist() == [1.0, 1.0, 0.0]


def test_zero_advantage_batch_has_finite_zero_gradients():
    current = torch.tensor([-2.0, -3.0], requires_grad=True)
    inputs = LossInputs(current, current.detach(), None, torch.zeros(2), torch.ones(2, dtype=torch.bool))
    result = clipped_grpo(inputs, 0.2)
    result.loss.backward()
    assert result.loss.item() == 0.0
    assert current.grad.tolist() == [0.0, 0.0]
    assert result.metrics["study/noncontributing_token_fraction"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("weighted", [False, True])
def test_noncontributing_metric_matches_autograd_at_bounds_and_underflow(weighted):
    bounds = torch.tensor([0.8, 1.2])
    ratios = torch.cat(
        [
            torch.tensor([0.5, 1.0, 1.5]),
            bounds,
            torch.nextafter(bounds, torch.zeros(2)),
            torch.nextafter(bounds, torch.full((2,), 2.0)),
        ]
    )
    values = torch.cat([ratios.log(), torch.tensor([-1000.0])])
    advantages = torch.tensor([-1.0, -0.875, -0.5, -0.125, 0.0, 0.125, 0.5, 0.875, 1.0])
    current = values.repeat_interleave(len(advantages)).requires_grad_()
    advantage = advantages.repeat(len(values))
    weights = torch.ones_like(current) if weighted else None
    if weighted:
        weights[::3] = 0
    mask = torch.ones_like(current, dtype=torch.bool)
    mask[0] = False
    result = clipped_grpo(LossInputs(current, torch.zeros_like(current), None, advantage, mask, weights), 0.2)
    result.loss.backward()
    assert torch.equal(result.metrics["study/noncontributing_token_fraction"].bool(), current.grad[mask] == 0)
    assert result.metrics["study/noncontributing_token_fraction"].requires_grad is False


def test_outside_bounds_does_not_imply_zero_training_contribution():
    current = torch.tensor([0.5, 1.5, 0.5, 1.5, 1.0]).log().requires_grad_()
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0, 0.0])
    result = clipped_grpo(
        LossInputs(current, torch.zeros_like(current), None, advantages, torch.ones(5, dtype=torch.bool)), 0.2
    )
    result.loss.backward()
    assert result.metrics["study/ratio_outside_clip_range"].mean().item() == pytest.approx(0.8)
    assert result.metrics["study/surrogate_clipped_fraction"].mean().item() == pytest.approx(0.4)
    assert result.metrics["study/noncontributing_token_fraction"].mean().item() == pytest.approx(0.6)
    assert (current.grad == 0).tolist() == [False, False, True, True, True]


@pytest.mark.parametrize("advantage", [0.0, 1.0, -1.0])
def test_overflow_is_rejected_before_an_optimizer_update(advantage):
    current = torch.tensor([0.0], requires_grad=True)
    inputs = LossInputs(current, torch.tensor([-100.0]), None, torch.tensor([advantage]), torch.tensor([True]))
    with pytest.raises(FloatingPointError):
        clipped_grpo(inputs, 0.2)
    assert current.grad is None
