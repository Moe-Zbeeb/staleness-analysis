# Cluster operations

The deployed project is `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study`. Job `2144915` installs its isolated Linux environment and runs asset checks, configuration resolution and the test suite on the CPU node `deep-chungus-6`. Job `2144916` depends on that job succeeding and requests an exclusive eight-A100 node for CUDA/NCCL, BF16 backward, Flash Attention backward and vLLM extension checks. These are preparation and diagnostic jobs, not a study run. Their effective submission commands are recorded in `diagnostics/` locally and `diagnostics/cluster/` remotely.

```bash
squeue -j 2144915,2144916
tail -n 50 diagnostics/cluster/setup-2144915.log
tail -n 50 diagnostics/cluster/gpu-health-2144916.log
```

Successful preparation writes `diagnostics/cluster/preparation.json`; successful GPU checks write `diagnostics/cluster/gpu-health-2144916.json`. A queued or running state does not establish success. The GPU check covers the runtime and collective topology; a short complete training-and-resume test and full-length memory checks are still required before claiming the study is validated end to end.

The CPU preparation job and GPU health check do not invoke `deepseek-study run`. Setup leaves training unlaunched. After choosing an actual lag and GPU profile, create a run configuration with `deepseek-study init`, inspect it with `build` and `check`, and launch `run` only within an authorized matching full-node Slurm allocation.
