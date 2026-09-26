import msgspec
import pytest
import verifiers.v1 as vf

from deepseek_study.algorithm import StudyGRPO
from deepseek_study.build import resolve
from deepseek_study.controller import Payload, payload_digest
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Progress
from prime_rl.transports.batch import TrainingSample
from prime_rl.trainer.batch import build_bin_cost, prepare_batch


def episode(group, reward):
    task = vf.TraceTask(type="Task", data=vf.TaskData(idx=group, prompt="problem"), key=f"problem-{group}")
    trace = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[
            vf.MessageNode(
                message=vf.UserMessage(content="problem"),
                token_ids=[10, 11],
                mask=[False, False],
                logprobs=[],
                sampled=False,
            ),
            vf.MessageNode(
                message=vf.AssistantMessage(content="answer"),
                token_ids=[12, 13],
                mask=[True, True],
                logprobs=[-1.0, -2.0],
                sampled=True,
                parent=0,
            ),
        ],
        calls=[vf.ModelCall(node=1, usage=vf.Usage(prompt_tokens=2, completion_tokens=2), finish_reason="stop")],
        rewards={"correctness": vf.Reward(score=reward)},
        ok=True,
    )
    return vf.Episode(
        task=task,
        env=vf.EnvInfo(id="deepseek-deepscaler", name="cleaned-deepscaler"),
        group=vf.GroupInfo(id=f"group-{group}"),
        run=vf.TrainRunInfo(id="test", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=0, end=0))),
        traces=[trace],
        ok=True,
    )


@pytest.mark.parametrize("all_zero", [False, True])
async def test_official_sink_keeps_entire_groups_and_behavior_logprobs(study, all_zero):
    config = resolve(study).orchestrator
    envs = TrainEnvs(config.train.source, config.env_addresses, clients=None)
    for env in envs:
        env.algorithm = StudyGRPO(env.config.algo, None, study)
    sink = TrainSink(config, tokenizer=None, train_envs=envs, progress=Progress(), batch_size=8, token_batch_size=None)
    batches = []
    for index in range(8):
        result = await sink.add(episode(index // 4, 0.0 if all_zero else float(index % 4 >= 2)))
        if result is not None:
            batches.append(result)
    assert len(batches) == 1
    samples = batches[0].samples
    assert len(samples) == 8
    encoded = msgspec.msgpack.encode(samples)
    restored = msgspec.msgpack.decode(encoded, type=list[TrainingSample])
    for index, sample in enumerate(restored):
        assert sample.mask == [False, False, True, True]
        assert sample.logprobs == [0.0, 0.0, -1.0, -2.0]
        expected = 0.0 if all_zero else (-1.0 if index % 4 < 2 else 1.0)
        assert sample.advantages[-2:] == pytest.approx([expected, expected])
        assert sample.temperatures == [1.0] * 4
        assert sample.ce_weights is None and sample.ref_kl_weights is None
    assert not sink.pending_batch and not sink.buffered_count()
    payload = Payload(encoded, ((0, 0),) * 8, (0.0,) * 8, (False,) * 8, ("task",) * 8)
    restored[0].logprobs[-1] = -20.0
    changed = Payload(
        msgspec.msgpack.encode(restored), payload.policy_spans, payload.rewards, payload.truncated, payload.task_keys
    )
    assert payload_digest(payload) != payload_digest(changed)


@pytest.mark.parametrize("workers", [1, 2, 4, 5])
def test_official_packing_preserves_every_loss_token_and_original_logprob(workers):
    samples = []
    expected = {}
    for index, length in enumerate([3, 5, 7, 10, 13]):
        tokens = [100 * (index + 1) + offset for offset in range(length)]
        mask = [False, False] + [True] * (length - 2)
        logprobs = [-0.5 - offset / 100 for offset in range(length)]
        advantages = [float(index - 2)] * length
        sample = TrainingSample(
            token_ids=tokens,
            mask=mask,
            logprobs=logprobs,
            advantages=advantages,
            temperatures=[1.0] * length,
            env_name="cleaned-deepscaler",
        )
        samples.append(sample)
        expected.update(
            {
                token: (logp, advantage)
                for token, train, logp, advantage in zip(tokens, mask, logprobs, advantages)
                if train
            }
        )
    grid = prepare_batch(
        samples, seq_len=16, num_train_workers=workers, bin_cost=build_bin_cost(None), pad_to_multiple_of=8
    )
    assert len(grid) == workers
    assert len({len(microbatches) for microbatches in grid}) == 1
    actual = {}
    for rank in grid:
        for batch in rank:
            for token, train, logp, advantage in zip(
                batch.input_ids, batch.loss_mask, batch.inference_logprobs, batch.advantages, strict=True
            ):
                if train:
                    assert token not in actual
                    actual[token] = (logp, advantage)
    assert actual == expected
