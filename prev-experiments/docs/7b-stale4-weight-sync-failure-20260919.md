# 7B staleness ≤4 weight-sync failure, 2026-09-19

## Finding

Job **2142033** failed on **deep-chungus-10** during an inference weight update after trainer step **238**. Its orchestrator's `WeightWatcher` task raised `httpx.ReadTimeout`; launcher cleanup subsequently terminated the trainer and inference processes. Slurm recorded `FAILED`, exit `1:0`, at **15:16:54 UTC**. The latest resumable paired trainer/orchestrator checkpoint remains **225**.

This was a runtime failure after successful restoration and steps 226–238. Increasing the existing 7,200-second startup weight wait would not address this failure. The captured trainer receiver wait reached 1,333 seconds, below its 1,800-second timeout; no trainer NCCL timeout expiration was captured.

The final traceback establishes an HTTP **POST to localhost:8574 with a 300-second read timeout**. It retains the `safe_cancel`/finished `WeightWatcher` frames but omits the endpoint-specific calling frames. Therefore `/pause` versus `/resume` is not directly proven. `/resume` after an unsuccessful weight update is the stronger inference from the sequence below; the fatal exception must not be described as a directly observed 720-second `/update_weights` timeout.

## Observed sequence

All times below are UTC. Application logs use EDT, four hours behind UTC; Rust router timestamps already use UTC.

| Time | Evidence |
| --- | --- |
| 13:21:23 | Trainer step 226 completed after restoring checkpoint 225. |
| 14:28:36 | Trainer step 237 completed. Orchestrator emitted rollout step 239, then waited for inference policy 238; its applied policy remained 237. |
| 14:31:06 | Inference began receiving the next 29 layer state dictionaries after a successful `/pause`. |
| 14:31:31–14:51:54 | A **20m23s gap** between receiving layer dictionaries 17 and 18. Orchestrator periodic logging also had a 20m23s gap, 14:31:41–14:52:04. |
| 14:52:08–14:52:10 | Inference reached dictionary 29; trainer step 238 completed. Inference immediately started another weight receive, without a subsequent layer-count log. No successful `/update_weights` or `/resume` response was logged for this update. |
| 14:54:57–15:16:10 | Trainer repeatedly reported waiting for the next broadcast receiver, from 60 to 1,333 seconds. |
| 15:13:56–15:14:00 | Orchestrator forced cleanup, wrote its own progress at step 239, logged the fatal timeout, and exited 1. Launcher began terminating the other processes. |
| 15:16:54 | Slurm recorded job failure. Trainer SIGTERM and the router's shutdown messages are consequences of cleanup. |

Checkpoint 239 contains only `orchestrator/progress.pt`; it has no trainer metadata or distributed checkpoint shards. It cannot resume training. Checkpoint 225 has the paired orchestrator progress, trainer `.metadata`, and all four nonempty trainer shards. Thirteen completed updates after 225 were not saved as a paired checkpoint.

## Likely mechanism and remaining uncertainty

The deployed source confirms a retry hazard:

- `orchestrator/clients.py:256–279` uses a 300-second admin read timeout, a 720-second weight-update timeout, and retries HTTP timeout/transport errors. `update_weights` acknowledges the trainer once, then invokes the retrying update POST and finally tries to resume inference.
- `inference/vllm/server.py:83–86` issues a new `update_weights_from_path` collective for every `/update_weights` POST. There is no request-version deduplication.
- `inference/vllm/worker/nccl.py` ignores the supplied checkpoint path when receiving: each invocation enters another NCCL receive. `transports/weights/nccl.py:214–220` supplies the one-time acknowledgement callback.

A slow first transfer can exceed the HTTP timeout while still running in the inference worker. Retrying can enqueue a second receive for the same update; after the first transfer completes, that receive waits for a broadcast while the trainer waits for acknowledgement of the next policy. The immediate second receive and unchanged orchestrator policy strongly support this explanation, but request identifiers were not logged, so the duplicated request is inferred rather than directly traced.

The long pauses correlate with the other job's checkpoint writes on the same host:

| 3B ≤6 save | Save interval | Overlapping 7B pause |
| --- | --- | --- |
| 825 | 13:32:53–13:52:48 | Orchestrator gaps of 13m05s and 5m29s; trainer step 230 took 20m02s. |
| 850 | 14:05:14–14:17:48 | Orchestrator gap of 11m31s; trainer waited for its receiver. |
| 875 | 14:30:53–14:52:00 | The 20m23s gap in both weight reception and orchestrator logging. |

Shared filesystem or host I/O pressure is a credible trigger, not yet a demonstrated kernel/storage cause. There is no captured CUDA OOM, disk-full error, fatal NCCL error, or node reboot. Repeated `RotaryEmbedding: Failed to load weights` warnings also occurred before successful updates, so that warning alone does not explain the failure.

At the 15:21 UTC status capture the node was `MIXED`, and **3B ≤6 job 2142032 was still running**: trainer step 899 had completed and checkpoint 900 was actively being written. Checkpoint 900 was not yet structurally complete; the preceding complete paired checkpoint was 875.

## Safe next step

Prepare and test an isolated recovery change that prevents duplicate NCCL receives for the same policy version, or fails safely without blindly replaying an uncertain update. Preserve the frozen experiment and record the recovery source/configuration separately. Test delayed HTTP responses with an in-flight collective, then validate several actual consecutive weight updates before resuming production from paired checkpoint 225. Align transfer/admin deadlines with the operation being waited on, and investigate the correlated checkpoint I/O stalls; extending the startup timeout alone is insufficient. No recovery implementation or restart was performed for this diagnosis.

## Recovery prepared after authorization

The [isolated recovery package](../recoveries/7b-stale4-weight-sync-20260919/README.md) now replaces only the job-local weight-update client. It sends each pause/update/resume request once, allows a 3,600-second read wait, and stops on an uncertain update without resuming inference or advancing the policy. All 32 local CPU tests and both shell syntax checks passed. Tests reproduce the original duplicate-request behavior and check the captured receiver/watcher sequence with the replacement.

The intended restart preserves paired checkpoint 225, optimizer/scheduler state, normal priority, and the original four-trainer/one-inference GPU topology on deep-chungus-10. Orchestrator-only checkpoint 239 is preserved separately. A GPU guard rejects any assigned GPU with a pre-existing compute process. The shared runtime and other jobs are not patched.

After the first Duo window timed out, the user authorized another SSH attempt, which succeeded. The package was deployed, cluster validation **2142051 passed**, and normal-priority five-GPU recovery **2142052** started on deep-chungus-10. At 18:19 UTC it had passed the runtime, experiment and five-GPU health checks; model staging and actual training progress remained pending. See the [recovery record](7b-stale4-weight-sync-recovery-20260919.md) for current evidence. Package manifest SHA-256: `0bd3e9f6d022ff2ab1a0c0126deedb57388434f2da28eb28633dec59a438ef5b`.

## Evidence and identity

- [Status capture](../../tmp/math-sweep-15b-normal-20260919/check_node10.json), observed 2026-09-19 15:21:10 UTC; SHA-256 `184610e8333b863a526d46603b0dab206dcab2650216131b8dad53e479082d86`.
- [Full orchestrator traceback, weight events and deployed source](../../tmp/math-sweep-15b-normal-20260919/check_node10_weight_sync.json), observed 2026-09-19 15:24:09 UTC; SHA-256 `e0ac2e88b30f71899ff9bd09785e02e94a540362689edd9886d445e55a3836a7`.
- [Recovery specification](../recoveries/7b-stale4-normal-retry-20260919/spec.json), SHA-256 `f128b0ccb8c2c049d52bde3505c47450d56df5eb87bff14268a228ad680d740d`.
- Prime-RL commit: `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`. Captured `orchestrator/clients.py` SHA-256: `a2d117b78b34c638cc19f3a92e8ec166634a23120319a14ae51ea570afa9fcc7`; `inference/vllm/server.py`: `2634995518bf7282908552a92cdad59fa0daa9a617697510dfe51234c753aa1b`; `inference/vllm/worker/nccl.py`: `627dd5678dc859a9455624e723a5e08f39a92257ce82a093a7a83310d81270bb`.
- Remote run: `/mnt/nfs/home/mohamadzbib/projects/rl-infra/outputs/dapo-qwen25-math7b-grpo-stale4-5gpu/qwen25-math7b-grpo-seed42-stale4-5gpu`; logs in `logs/attempt_4`.
