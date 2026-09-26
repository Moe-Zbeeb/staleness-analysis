import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepseek_study.learning.loss import clipped_grpo
from deepseek_study.tracking.archive import MetricMirror, reserve_mirror, sha256
from deepseek_study.tracking.evaluation import SUITES, import_evaluation
from deepseek_study.tracking.observer import observe_papers
from deepseek_study.tracking.paper import read_tokens, summarize
from deepseek_study.tracking.runboard import Metrics
from deepseek_study.tracking.tokens import PaperTokenExporter, setup_paper_exporter
from prime_rl.trainer.rl.loss import LossInputs


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def shard(output, rank, count=4, step=33):
    current = torch.tensor([99.0] + [-1.0, -3.0, -2.0, -2.0] * count, requires_grad=True)
    behavior = torch.tensor([-99.0] + [-2.0] * 4 * count)
    advantage = torch.tensor([99.0] + [1.0, -1.0, 0.0, -1.0] * count)
    mask = torch.tensor([False] + [True] * 4 * count)
    inputs = LossInputs(current, behavior, None, advantage, mask)
    loss = clipped_grpo(inputs, 0.2).loss
    loss.backward()
    gradient = current.grad.clone()
    exporter = PaperTokenExporter(output, rank, 0.2)
    batch = {
        "loss_mask": mask.reshape(1, -1),
        "input_ids": torch.arange(len(current)).reshape(1, -1),
        "position_ids": torch.arange(len(current)).reshape(1, -1),
        "advantages": advantage.reshape(1, -1),
        "inference_logprobs": behavior.reshape(1, -1),
    }
    entropy = torch.tensor([999.0] + [1.0, 2.0, 3.0, 4.0] * count)
    exporter.export(step, 0, batch, {"logprobs": current, "entropy": entropy}, [2] + [1] * (len(current) - 2), None)
    exporter.mark_stable()
    assert torch.equal(current.grad, gradient)
    with np.load(output / f"paper/tokens/step_{step}/rank_{rank}.npz") as saved:
        np.testing.assert_array_equal(saved["zero_policy_signal"], current.grad[mask].numpy() == 0)
    return current, behavior, advantage


def test_token_export_preserves_gradient_and_uses_complete_rank_union(tmp_path):
    shard(tmp_path, 0, count=1)
    shard(tmp_path, 1, count=3)
    arrays = read_tokens(tmp_path / "paper/tokens/step_33", 2)
    assert len(arrays["current_logp"]) == 16
    assert arrays["entropy"].mean() == 2.5
    assert arrays["current_logp"].max() == -1
    metrics, frequency = summarize(arrays, 0.2)
    assert metrics["clip/fraction"] == 0.5
    assert metrics["gradient_signal/noncontributing_token_fraction"] == 0.75
    assert metrics["gradient_signal/contributing_token_fraction"] == 0.25
    assert metrics["gradient_signal/noncontributing_tokens"] == 12
    assert metrics["gradient_signal/zero_advantage_fraction"] == 0.25
    assert metrics["gradient_signal/clipped_zero_gradient_fraction"] == 0.5
    assert metrics["gradient_signal/numerical_zero_fraction"] == 0
    assert metrics["gradient_signal/recorded_mask_fraction"] == 1
    assert metrics["clip/outside_range_fraction"] == 0.5
    assert metrics["mismatch/m2"] == 0.5
    assert metrics["mismatch/kl_k1"] == 0
    assert metrics["entropy/mean"] == 2.5
    assert metrics["loss/grpo_global_token_mean"] == pytest.approx((-1.2 + 0.8 + 1) / 4)
    assert metrics["advantage/positive/token_fraction"] == 0.25
    assert metrics["advantage/negative/token_fraction"] == 0.5
    assert metrics["advantage/zero/token_fraction"] == 0.25
    assert metrics["length/all/count"] == 16
    assert sum(frequency["counts"]) == 8
    assert metrics["joint/entropy_by_ratio_distance/bin_00/count"] == 8
    assert metrics["joint/entropy_by_ratio_distance/bin_00/mean"] == 3.5
    with pytest.raises(FileExistsError):
        shard(tmp_path, 0, count=1)


def test_global_reduction_is_not_average_of_rank_means(tmp_path):
    shard(tmp_path, 0, count=1)
    shard(tmp_path, 1, count=3)
    path = tmp_path / "paper/tokens/step_33/rank_1.npz"
    with np.load(path) as data:
        arrays = dict(data)
    arrays["entropy"][:] = 10
    arrays["current_logp"][:] = arrays["behavior_logp"]
    arrays["advantage"][:] = 1
    arrays["surrogate_clipped"][:] = False
    arrays["zero_policy_signal"][:] = False
    np.savez_compressed(path, **arrays)
    metrics, _ = summarize(read_tokens(path.parent, 2), 0.2)
    assert metrics["entropy/mean"] == (2.5 * 4 + 10 * 12) / 16
    assert metrics["entropy/mean"] != (2.5 + 10) / 2
    assert metrics["gradient_signal/noncontributing_token_fraction"] == 3 / 16
    assert metrics["gradient_signal/noncontributing_token_fraction"] != (0.75 + 0) / 2


def test_empty_advantage_subsets_are_absent_not_false_zeros(tmp_path):
    shard(tmp_path, 0, count=1)
    arrays = read_tokens(tmp_path / "paper/tokens/step_33", 1)
    arrays["advantage"][:] = 0
    arrays["surrogate_clipped"][:] = False
    arrays["zero_policy_signal"][:] = True
    metrics, _ = summarize(arrays, 0.2)
    assert metrics["loss/grpo_global_token_mean"] == 0
    assert metrics["clip/fraction"] == 0
    assert metrics["gradient_signal/noncontributing_token_fraction"] == 1
    assert metrics["gradient_signal/contributing_token_fraction"] == 0
    assert "loss/positive/absolute_share" not in metrics
    assert "advantage/positive/entropy_mean" not in metrics
    assert "length/positive/mean" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())


def test_exporter_rejects_duplicate_data_parallel_or_context_parallel_layout(tmp_path):
    config = SimpleNamespace(enable_token_export=True, output_dir=tmp_path)
    dims = SimpleNamespace(cp_enabled=True, get_mesh=lambda _: SimpleNamespace(size=lambda: 2))
    with pytest.raises(ValueError, match="unique data shard"):
        setup_paper_exporter(config, dims, SimpleNamespace(world_size=2), SimpleNamespace())


def test_paper_observer_only_processes_completed_updates_and_replays_idempotently(study, tmp_path):
    study = study.model_copy(
        update={
            "trainer_gpus": 2,
            "prompts_per_update": 2,
            "responses_per_prompt": 4,
            "metrics_mirror_root": tmp_path / "xfs",
        }
    )
    output = study.output_dir
    save(output / "configs/study.json", study.model_dump(mode="json"))
    save(output / "run.json", {"run_uuid": "unique"})
    shard(output, 0, count=1)
    shard(output, 1, count=1)
    shard(output, 0, count=1, step=34)
    receipt = {"step": 33, "learner_version": 32, "behavior_version": 0, "warmup": False}
    (output / "updates.jsonl").write_text(json.dumps(receipt) + "\n")
    save(output / "run-status.json", {"status": "finished"})
    result = observe_papers(output, once=True)
    assert result["steps"] == 1
    metrics = output / "paper-metrics.jsonl"
    original = metrics.read_bytes()
    assert json.loads(original)["step"] == 33
    assert json.loads(original)["metrics"]["clip/run_update_mean"] == 0.5
    assert observe_papers(output, once=True)["steps"] == 1
    assert metrics.read_bytes() == original
    assert not (output / "paper/steps/34.json").exists()
    mirror = study.metrics_mirror_root / output.name
    manifest = json.loads((mirror / "mirror-manifest.json").read_text())["verified_files"]
    assert manifest["paper-metrics.jsonl"]["sha256"] == sha256(metrics)
    assert (mirror / "paper/tokens/step_33/rank_0.npz").read_bytes() == (
        output / "paper/tokens/step_33/rank_0.npz"
    ).read_bytes()
    assert not (mirror / "checkpoints").exists()
    rows = []
    observer = Metrics(output, SimpleNamespace(log=lambda data, step: rows.append((step, data))))
    observer.poll()
    assert any(row[1].get("paper/mismatch/m2") == 0.5 for row in rows)
    assert any(row[1].get("paper/gradient_signal/noncontributing_token_fraction") == 0.75 for row in rows)
    assert manifest["paper/tokens/step_33/rank_0.npz"]["sha256"] == sha256(output / "paper/tokens/step_33/rank_0.npz")


def test_legacy_raw_tokens_reconstruct_signal_without_claiming_recorded_masks(tmp_path):
    shard(tmp_path, 0, count=1)
    path = tmp_path / "paper/tokens/step_33/rank_0.npz"
    with np.load(path) as data:
        arrays = {key: value for key, value in data.items() if key not in {"surrogate_clipped", "zero_policy_signal"}}
    np.savez_compressed(path, **arrays)
    receipt = json.loads(path.with_suffix(".json").read_text())
    receipt["schema_version"] = 1
    save(path.with_suffix(".json"), receipt)
    metrics, _ = summarize(read_tokens(path.parent, 1), 0.2)
    assert metrics["gradient_signal/noncontributing_token_fraction"] == 0.75
    assert metrics["gradient_signal/recorded_mask_fraction"] == 0


def test_mirror_rejects_wrong_owner_and_corrupt_prefix(tmp_path):
    output = tmp_path / "run"
    save(output / "run.json", {"run_uuid": "first"})
    (output / "updates.jsonl").write_text('{"step":1}\n')
    mirror = MetricMirror(output, tmp_path / "xfs")
    mirror.poll()
    (mirror.destination / "updates.jsonl").write_text('{"step":2}\n')
    with pytest.raises(ValueError, match="prefix"):
        MetricMirror(output, tmp_path / "xfs").poll()
    save(output / "run.json", {"run_uuid": "second"})
    with pytest.raises(FileExistsError, match="another run"):
        reserve_mirror(output, tmp_path / "xfs")


def eval_fixture(study, tmp_path, suite="bapo"):
    output = study.output_dir
    save(output / "configs/study.json", study.model_dump(mode="json"))
    save(output / "run-status.json", {"status": "finished"})
    save(output / "checkpoints/step_100/study/complete.json", {"step": 100, "components_sha256": "checkpoint"})
    settings = {
        "suite": suite,
        "grader_identity": "test",
        "sampling": {"temperature": 1},
        "prompt_template_sha256": "prompt",
        "checkpoint_components_sha256": "checkpoint",
        "benchmarks": {},
    }
    rows = []
    for name in SUITES[suite]:
        settings["benchmarks"][name] = {
            "question_ids": ["q"],
            "samples_per_question": 16,
            "dataset_revision": "test",
            "dataset_sha256": "test",
        }
        rows.extend(
            {"benchmark": name, "question_id": "q", "repeat": repeat, "correct": repeat < 8} for repeat in range(16)
        )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    protocol = tmp_path / "protocol.json"
    save(protocol, settings)
    return output, predictions, protocol


@pytest.mark.parametrize("suite", list(SUITES))
def test_offline_evaluation_uses_mean_correctness_not_pass_at_16(study, tmp_path, suite):
    output, predictions, protocol = eval_fixture(study, tmp_path, suite)
    result = import_evaluation(output, predictions, protocol, 100)
    assert result["metrics"][f"{suite}/macro_accuracy"] == 0.5
    assert result["metrics"][f"{suite}/macro_accuracy_percent"] == 50
    import_evaluation(output, predictions, protocol, 100)
    assert len((output / "evaluation-metrics.jsonl").read_text().splitlines()) == 1


def test_offline_evaluation_rejects_missing_samples_and_wrong_checkpoint(study, tmp_path):
    output, predictions, protocol = eval_fixture(study, tmp_path)
    predictions.write_text("\n".join(predictions.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing or unexpected"):
        import_evaluation(output, predictions, protocol, 100)
    settings = json.loads(protocol.read_text())
    settings["checkpoint_components_sha256"] = "wrong"
    save(protocol, settings)
    with pytest.raises(ValueError, match="checkpoint identity"):
        import_evaluation(output, predictions, protocol, 100)
