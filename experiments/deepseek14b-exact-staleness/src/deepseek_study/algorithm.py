import torch

from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages


def group_advantages(rewards, normalization, epsilon):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("GRPO needs a complete group with finite rewards")
    advantages = values - values.mean()
    if normalization != "centered":
        if normalization not in {"population_std", "sample_std"}:
            raise ValueError("Unknown advantage normalization")
        advantages = advantages / (values.std(correction=int(normalization == "sample_std")) + epsilon)
    return advantages


class StudyGRPO(Algorithm):
    def __init__(self, config, clients, study):
        super().__init__(config, clients)
        self.study = study

    async def score_group(self, episodes):
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if len(episodes) != self.study.responses_per_prompt or len(traces) != len(episodes):
            raise ValueError("Missing, failed, or multiple traces in a GRPO group")
        advantages = group_advantages(
            [trace.reward for trace in traces],
            self.study.advantage_normalization,
            self.study.advantage_epsilon,
        )
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
