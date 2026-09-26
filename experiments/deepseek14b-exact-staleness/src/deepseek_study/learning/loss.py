import torch

from prime_rl.trainer.rl.loss import LossOutputs


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
    if inputs.loss_weights is not None:
        objective = objective * inputs.loss_weights[mask]
    return LossOutputs(
        loss=-objective.sum(),
        metrics={
            "study/ratio": ratio.detach(),
            "study/ratio_outside_clip_range": ((ratio - 1).abs() > clip_epsilon).float().detach(),
            "study/surrogate_clipped_fraction": (
                ((advantages > 0) & (ratio > 1 + clip_epsilon)) | ((advantages < 0) & (ratio < 1 - clip_epsilon))
            )
            .float()
            .detach(),
            "study/behavior_log_ratio": (current - behavior).detach(),
        },
    )
