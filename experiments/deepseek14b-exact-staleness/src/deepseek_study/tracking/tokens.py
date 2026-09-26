import json
import os
from pathlib import Path

import numpy as np
import torch

from deepseek_study.learning.loss import token_signal_masks
from deepseek_study.runtime.checkpoints import atomic_write


COLUMNS = (
    "token_id",
    "position",
    "current_logp",
    "behavior_logp",
    "advantage",
    "entropy",
    "surrogate_clipped",
    "zero_policy_signal",
)


class PaperTokenExporter:
    def __init__(self, output, rank, clip_epsilon):
        self.output = Path(output) / "paper" / "tokens"
        self.rank = rank
        self.clip_epsilon = clip_epsilon
        self.step = None
        self.parts = {key: [] for key in COLUMNS}
        self.lengths = []

    @torch.no_grad()
    def export(self, step, micro_step, micro_batch, model_output, sequence_lengths, loss_config):
        if self.step not in (None, step):
            raise RuntimeError("Previous paper token step was not sealed")
        self.step = step
        mask = micro_batch["loss_mask"].detach().cpu().reshape(-1).numpy().astype(bool)
        if sum(sequence_lengths) != len(mask):
            raise ValueError("Paper export sequence boundaries do not match the packed batch")
        weights = micro_batch.get("rl_weights")
        if weights is not None and not torch.all(weights.reshape(-1).cpu()[torch.from_numpy(mask)] == 1):
            raise ValueError("Paper diagnostics require the study's unweighted GRPO tokens")
        tensors = {
            "token_id": micro_batch["input_ids"],
            "position": micro_batch["position_ids"],
            "current_logp": model_output["logprobs"],
            "behavior_logp": micro_batch["inference_logprobs"],
            "advantage": micro_batch["advantages"],
            "entropy": model_output["entropy"],
        }
        device_mask = micro_batch["loss_mask"].reshape(-1).bool()
        current = model_output["logprobs"].reshape(-1)[device_mask].float()
        behavior = micro_batch["inference_logprobs"].reshape(-1)[device_mask].float()
        advantages = micro_batch["advantages"].reshape(-1)[device_mask].float()
        clipped, no_signal = token_signal_masks(torch.exp(current - behavior), advantages, self.clip_epsilon)
        self.parts["surrogate_clipped"].append(clipped.cpu().numpy().copy())
        self.parts["zero_policy_signal"].append(no_signal.cpu().numpy().copy())
        for key, tensor in tensors.items():
            dtype = torch.int32 if key in {"token_id", "position"} else torch.float32
            values = tensor.detach().to(device="cpu", dtype=dtype).reshape(-1).numpy()
            if len(values) != len(mask):
                raise ValueError("Misaligned paper token column")
            self.parts[key].append(values[mask].copy())
        start = 0
        for length in sequence_lengths:
            count = int(mask[start : start + length].sum())
            if count:
                self.lengths.append(count)
            start += length

    def mark_stable(self):
        if self.step is None:
            raise RuntimeError("Cannot seal an empty paper export step")
        directory = self.output / f"step_{self.step}"
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.concatenate(parts) for key, parts in self.parts.items()}
        arrays["response_lengths"] = np.asarray(self.lengths, dtype=np.int32)
        destination = directory / f"rank_{self.rank}.npz"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite token evidence: {destination}")
        temporary = destination.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        atomic_write(
            directory / f"rank_{self.rank}.json",
            json.dumps(
                {"schema_version": 2, "step": self.step, "rank": self.rank, "tokens": len(arrays["token_id"])}
            ).encode(),
        )
        self.step = None
        self.parts = {key: [] for key in COLUMNS}
        self.lengths = []

    def close(self):
        return


def setup_paper_exporter(config, parallel_dims, world, logger):
    if (
        not config.enable_token_export
        or parallel_dims.cp_enabled
        or parallel_dims.get_mesh("dp").size() != world.world_size
    ):
        raise ValueError("Paper exporter requires enabled token export and one unique data shard per trainer rank")
    logger.info("Archiving detached paper diagnostics from the existing forward pass")
    return PaperTokenExporter(config.output_dir, world.rank, config.loss.kwargs["clip_epsilon"])
