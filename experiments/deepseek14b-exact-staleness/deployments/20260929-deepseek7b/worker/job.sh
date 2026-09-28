#!/bin/bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset NCCL_NET NCCL_IB_DISABLE NCCL_SOCKET_FAMILY NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME VLLM_HOST_IP DEEPSEEK_STUDY_REMOTE_INFERENCE
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
result=0
/usr/bin/python3 /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/worker/launch_historical_worker.py --control /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/worker --learner-job "$1" || result=$?
mkdir -p /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/worker/worker-logs
rsync -a --include="logs/***" --include="source/***" --include="configs/***" --include="*.json" --include="*.jsonl" --exclude="*" /tmp/mohamadzbib-staleness-storage-v2/deepseek7b-dapo17k-6k-exact256-b32-seed42-v1-historical/runs/deepseek7b-dapo17k-6k-exact256-b32-seed42-v1-historical/ /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/worker/worker-logs/ || true
exit "$result"
