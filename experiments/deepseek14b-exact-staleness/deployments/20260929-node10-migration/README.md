# Node10 migration preparation, September 29

The requested DeepSeek-R1-Distill-Qwen-1.5B k=256 move was **not executed**. While the node-local runtime was being staged, another user queued job 2145792 for all eight GPUs and all 128 CPUs on node 10 at higher scheduler priority. That job subsequently started on node 10. Stopping the original job would have released our HP quota without guaranteeing a destination allocation.

Original job 2145464 continued on node 5 in high-priority partition/QoS. The final source check showed update 421 and exact age 256. No stop watcher was armed, no checkpoint was modified, and no optimizer or training process was interrupted. Existing k=0 and queued 7B jobs were unchanged.

Preparation jobs 2145791,2145793,2145794 and the held migration job 2145795 were cancelled. The node 10 runtime copy is partial and unsealed; it must not be used as a validated launch environment. Frozen source and prepared migration controls remain on shared storage at the control path in [status.json](status.json).

The implemented external controls support an audited four-trainer/five-inference to four-trainer/four-inference migration. They preserve scientific settings and full recovery state, validate the original shared checkpoint, copy it without changing payload bytes, and rebind only the copied completion marker's configuration hash. Existing frozen training source, PrimeRL and vLLM remain unchanged. Changing inference replicas can affect future sampling; the trajectory is not claimed to be bitwise identical.

Validation: 703 local tests passed, 4 were skipped; all 29 checkpoint-stop tests passed on Linux inside the original allocation. The local suite includes adapter-to-staging-to-runtime queue restoration. No destination GPU startup, resumed learner update or speedup was measured.

A later attempt must recheck destination availability and quota, finish and verify staging, select a fresh scheduled checkpoint boundary, verify final shared backup completion, audit/copy the checkpoint, and validate GPU health and the first resumed exact-age update. No cancelled job or old checkpoint should be blindly requeued as the source continues advancing.
