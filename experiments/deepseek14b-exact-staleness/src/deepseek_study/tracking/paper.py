import json
from pathlib import Path

import numpy as np

from deepseek_study.runtime.checkpoints import atomic_write
from deepseek_study.tracking.tokens import COLUMNS


RATIO_EDGES = np.asarray([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, np.inf])
DISTANCE_EDGES = np.r_[np.arange(0, 0.82, 0.02), 1, 2, np.inf]
PROBABILITY_EDGES = np.r_[np.linspace(0, 1, 21), np.inf]


def read_tokens(directory, ranks):
    shards = []
    for rank in range(ranks):
        path = Path(directory) / f"rank_{rank}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if receipt["schema_version"] != 1 or receipt["rank"] != rank or path.parent.name != f"step_{receipt['step']}":
            raise ValueError("Invalid token shard receipt")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in (*COLUMNS, "response_lengths")}
        count = receipt["tokens"]
        if any(len(arrays[key]) != count for key in COLUMNS) or arrays["response_lengths"].sum() != count:
            raise ValueError("Token columns or response lengths do not align")
        if any(not np.isfinite(arrays[key]).all() for key in COLUMNS):
            raise ValueError("Nonfinite token diagnostic; refusing to hide invalid values")
        if (arrays["response_lengths"] <= 0).any():
            raise ValueError("Invalid response token count")
        shards.append(arrays)
    return {key: np.concatenate([shard[key] for shard in shards]) for key in (*COLUMNS, "response_lengths")}


def summarize(arrays, epsilon):
    current = arrays["current_logp"].astype(np.float64)
    behavior = arrays["behavior_logp"].astype(np.float64)
    advantage = arrays["advantage"].astype(np.float64)
    entropy = arrays["entropy"].astype(np.float64)
    if not len(current):
        raise ValueError("No valid response tokens in optimizer update")
    log_ratio = current - behavior
    ratio = np.exp(log_ratio)
    probability = np.exp(current)
    behavior_probability = np.exp(behavior)
    if not np.isfinite(ratio).all():
        raise ValueError("Nonfinite importance ratios")
    positive, negative, zero = advantage > 0, advantage < 0, advantage == 0
    clipped = (positive & (ratio > 1 + epsilon)) | (negative & (ratio < 1 - epsilon))
    active = ~clipped & ~zero
    objective = np.minimum(ratio * advantage, np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)
    metrics = {}

    def mean(name, values):
        if len(values):
            metrics[name] = float(np.mean(values))

    def divide(name, numerator, denominator):
        if denominator > 0:
            metrics[name] = float(numerator / denominator)

    def stats(prefix, values):
        metrics[f"{prefix}/count"] = len(values)
        if len(values):
            mean(f"{prefix}/mean", values)
            metrics[f"{prefix}/std"] = float(np.std(values))
            for label, quantile in (
                ("min", 0),
                ("p25", 0.25),
                ("p50", 0.5),
                ("p75", 0.75),
                ("p80", 0.8),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1),
            ):
                metrics[f"{prefix}/{label}"] = float(np.quantile(values, quantile))

    metrics["tokens/count"] = len(ratio)
    metrics["clip/lower_bound"] = 1 - epsilon
    metrics["clip/upper_bound"] = 1 + epsilon
    mean("clip/fraction", clipped)
    mean("clip/outside_range_fraction", (ratio < 1 - epsilon) | (ratio > 1 + epsilon))
    mean("loss/grpo_global_token_mean", -objective)
    stats("entropy", entropy)
    stats("ratio", ratio)
    stats("log_ratio", log_ratio)
    mean("mismatch/kl_k1", -log_ratio)
    mean("mismatch/kl_absolute", np.abs(log_ratio))
    mean("mismatch/kl_k3", np.expm1(log_ratio) - log_ratio)
    mean("mismatch/m2", log_ratio**2)
    mean("mismatch/chi2_sample", (ratio - 1) ** 2)
    divide("mismatch/ess_fraction", ratio.sum() ** 2, len(ratio) * (ratio**2).sum())
    mean("mismatch/m2_active", log_ratio[active] ** 2)
    mean("mismatch/m2_active_global_denominator", log_ratio**2 * active)
    mean("entropy/clipped_mean", entropy[clipped])
    mean("entropy/active_mean", entropy[active])
    high_entropy = entropy >= np.quantile(entropy, 0.8)
    divide("clip/high_entropy_fraction", (clipped & high_entropy).sum(), high_entropy.sum())
    divide("clip/high_entropy_share", (clipped & high_entropy).sum(), clipped.sum())
    mean(
        "entropy_clip/empirical_covariance",
        (current - current.mean()) * (advantage * active - (advantage * active).mean()),
    )
    positive_mass = objective[positive].sum()
    negative_mass = -objective[negative].sum()
    for name, selection in (("positive", positive), ("negative", negative), ("zero", zero)):
        metrics[f"advantage/{name}/tokens"] = int(selection.sum())
        mean(f"advantage/{name}/token_fraction", selection)
        mean(f"advantage/{name}/clip_fraction", clipped[selection])
        mean(f"advantage/{name}/entropy_mean", entropy[selection])
        mean(f"advantage/{name}/m2", log_ratio[selection] ** 2)
        mean(f"advantage/{name}/kl_k1", -log_ratio[selection])
        mean(f"advantage/{name}/mean", advantage[selection])
        mean(f"loss/{name}/global_token_contribution", -objective * selection)
        mass = np.abs(objective[selection]).sum()
        divide(f"loss/{name}/absolute_share", mass, positive_mass + negative_mass)
        mean(f"gradient_signal/{name}/absolute_coefficient", np.abs(ratio * advantage) * active * selection)
    divide("loss/positive_to_negative_ratio", positive_mass, negative_mass)
    weighted = behavior_probability * objective
    divide("loss/bapo_equation8_empirical_ratio", np.abs(weighted[positive].sum()), np.abs(weighted.sum()))
    lengths = arrays["response_lengths"]
    offsets = np.r_[0, np.cumsum(lengths)]
    response_advantages = advantage[offsets[:-1]]
    for start, end, value in zip(offsets[:-1], offsets[1:], response_advantages, strict=True):
        if not np.all(advantage[start:end] == value):
            raise ValueError("A packed sequence is not a constant-advantage response")
    for name, selection in (
        ("all", np.ones(len(lengths), dtype=bool)),
        ("positive", response_advantages > 0),
        ("negative", response_advantages < 0),
        ("zero", response_advantages == 0),
    ):
        stats(f"length/{name}", lengths[selection])
        divide(f"responses/{name}/fraction", selection.sum(), len(lengths))

    def bins(prefix, x, edges, selection, y, quantiles=False):
        indices = np.searchsorted(edges, x, side="right") - 1
        for index in range(len(edges) - 1):
            mask = selection & (indices == index)
            base = f"{prefix}/bin_{index:02d}"
            metrics[f"{base}/count"] = int(mask.sum())
            mean(f"{base}/x_mean", x[mask])
            mean(f"{base}/mean", y[mask])
            if quantiles and mask.any():
                for q in (0, 25, 50, 75, 100):
                    metrics[f"{base}/p{q:02d}"] = float(np.percentile(y[mask], q))

    all_tokens = np.ones(len(ratio), dtype=bool)
    bins("joint/entropy_by_ratio_distance", np.abs(ratio - 1), DISTANCE_EDGES, all_tokens, entropy)
    bins("joint/entropy_by_probability", probability, PROBABILITY_EDGES, all_tokens, entropy)
    for name, selection in (("positive", positive), ("negative", negative), ("all", all_tokens)):
        bins(f"joint/{name}/probability_by_ratio", ratio, RATIO_EDGES, selection, probability, quantiles=True)
    vocab, counts = np.unique(arrays["token_id"][clipped], return_counts=True)
    order = np.lexsort((vocab, -counts))
    top = [{"token_id": int(vocab[i]), "count": int(counts[i])} for i in order[:50]]
    for item in top:
        metrics[f"clipped_tokens/id_{item['token_id']}/count"] = item["count"]
    if any(not np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite derived metric")
    return metrics, {"token_ids": vocab.tolist(), "counts": counts.tolist(), "top50": top}


def process_step(output, receipt, study):
    output = Path(output)
    step = receipt["step"]
    target = output / "paper" / "steps" / f"{step}.json"
    if target.is_file():
        existing = json.loads(target.read_text())
        if existing.get("config_sha256") != study.fingerprint():
            raise ValueError("Cached paper metrics use a different study configuration")
        return existing
    arrays = read_tokens(output / "paper" / "tokens" / f"step_{step}", study.trainer_gpus)
    if len(arrays["response_lengths"]) != study.response_batch_size:
        raise ValueError("Paper archive does not contain the complete training cohort")
    metrics, frequencies = summarize(arrays, study.clip_epsilon)
    record = {
        "schema_version": 1,
        "config_sha256": study.fingerprint(),
        "step": step,
        "learner_version": receipt["learner_version"],
        "behavior_version": receipt["behavior_version"],
        "bootstrap": receipt["warmup"],
        "metrics": metrics,
        "clipped_token_frequencies": frequencies,
    }
    atomic_write(target, json.dumps(record, allow_nan=False).encode())
    return record
