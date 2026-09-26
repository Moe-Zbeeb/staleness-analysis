import torch

from prime_rl.trainer.rl.loss import LossOutputs


@torch.no_grad()
def token_signal_masks(ratio, advantages, clip_epsilon, weights=None):
    raw = ratio * advantages
    bounded = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    ties = raw == bounded
    incoming = torch.ones_like(advantages) if weights is None else weights
    raw_gradient = torch.where(raw < bounded, incoming, torch.where(ties, incoming / 2, 0)) * advantages
    bounded_gradient = torch.where(raw > bounded, incoming, torch.where(ties, incoming / 2, 0)) * advantages
    inside = (ratio >= 1 - clip_epsilon) & (ratio <= 1 + clip_epsilon)
    coefficient = (raw_gradient + torch.where(inside, bounded_gradient, 0)) * ratio
    clipped = ((advantages > 0) & (ratio > 1 + clip_epsilon)) | ((advantages < 0) & (ratio < 1 - clip_epsilon))
    return clipped, coefficient == 0


def clipped_grpo(inputs, clip_epsilon):
    mask = inputs.loss_mask
    current = inputs.trainer_logprobs[mask].float()
    behavior = inputs.inference_logprobs[mask].detach().float()
    advantages = inputs.advantages[mask].detach().float()
    ratio = torch.exp(current - behavior)
    if not torch.isfinite(ratio).all() or not torch.isfinite(advantages).all():
        raise FloatingPointError("Nonfinite GRPO ratio or advantage; refusing a corrupted optimizer update")
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    weights = inputs.loss_weights[mask] if inputs.loss_weights is not None else None
    if inputs.loss_weights is not None:
        objective = objective * weights
    surrogate_clipped, no_signal = token_signal_masks(ratio.detach(), advantages, clip_epsilon, weights)
    return LossOutputs(
        loss=-objective.sum(),
        metrics={
            "study/ratio": ratio.detach(),
            "study/ratio_outside_clip_range": ((ratio - 1).abs() > clip_epsilon).float().detach(),
            "study/surrogate_clipped_fraction": surrogate_clipped.float(),
            "study/noncontributing_token_fraction": no_signal.float(),
            "study/behavior_log_ratio": (current - behavior).detach(),
        },
    )
