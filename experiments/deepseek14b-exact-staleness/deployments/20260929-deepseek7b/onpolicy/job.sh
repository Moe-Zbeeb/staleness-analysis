#!/bin/bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset NCCL_NET NCCL_IB_DISABLE NCCL_SOCKET_FAMILY NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME VLLM_HOST_IP DEEPSEEK_STUDY_REMOTE_INFERENCE
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/python3 /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/onpolicy/node_local_run.py stage --control /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/onpolicy
exec /usr/bin/python3 /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/onpolicy/node_local_run.py run --control /mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/launches/deepseek7b-dapo17k-b32-hp-20260929-v1/onpolicy
