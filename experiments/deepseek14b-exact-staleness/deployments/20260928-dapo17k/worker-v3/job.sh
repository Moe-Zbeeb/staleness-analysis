#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset NCCL_NET NCCL_IB_DISABLE NCCL_SOCKET_FAMILY NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME VLLM_HOST_IP DEEPSEEK_STUDY_REMOTE_INFERENCE
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
result=0
/usr/bin/python3 /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-hp-20260928-v1/worker-v3/launch_historical_worker.py --control /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-hp-20260928-v1/worker-v3 --learner-job "$1" || result=$?
mkdir -p /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-hp-20260928-v1/worker-v3/worker-logs
rsync -a --include="logs/***" --include="source/***" --include="configs/***" --include="*.json" --include="*.jsonl" --exclude="*" /tmp/mohamadzbib-staleness-storage-v2/deepseek15b-dapo17k-6k-exact256-seed42-v1-historical/runs/deepseek15b-dapo17k-6k-exact256-seed42-v1-historical/ /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek15b-dapo17k-async-hp-20260928-v1/worker-v3/worker-logs/ || true
exit "$result"
