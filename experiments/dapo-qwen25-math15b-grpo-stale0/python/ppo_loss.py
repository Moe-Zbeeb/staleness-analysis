import torch

from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * inputs.advantages, clipped * inputs.advantages)
    per_token_loss = -surrogate
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss[inputs.loss_mask].sum()
    selected = ratio[inputs.loss_mask]
    if selected.numel() == 0:
        return LossOutputs(loss=loss, metrics={})
    metrics = {
        "clip_fraction": (selected != clipped[inputs.loss_mask]).float().detach(),
        "importance_ratio": selected.detach(),
        "approx_kl": (selected - log_ratio[inputs.loss_mask] - 1).detach(),
    }
    return LossOutputs(loss=loss, metrics=metrics)
