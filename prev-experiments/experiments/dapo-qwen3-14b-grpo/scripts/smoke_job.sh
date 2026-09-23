set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen3-14b-grpo
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
MODEL_SOURCE=/mnt/nfs/home/mohamadzbib/projects/models/Qwen3-14B-40c06982
source "$RL_INFRA/env.sh" prime-rl
export PYTHONPATH="$EXPERIMENT_ROOT/python"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
test "$(git -C "$RL_INFRA/repos/prime-rl" rev-parse HEAD)" = ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
test -z "$(git -C "$RL_INFRA/repos/prime-rl" status --porcelain)"
PYTHONPATH="$EXPERIMENT_ROOT/python" uv run --no-sync python "$EXPERIMENT_ROOT/scripts/validate_experiment.py" --experiment-root "$EXPERIMENT_ROOT"
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
df -h /dev/shm
printenv NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
nvidia-smi --query-gpu=index,name,uuid,memory.total,temperature.gpu,power.draw --format=csv
uv run --no-sync torchrun --standalone --nproc-per-node=9 "$EXPERIMENT_ROOT/scripts/health_probe.py"
NODE_CACHE=${SLURM_TMPDIR:-/tmp/prime-rl-$SLURM_JOB_ID}
RUNTIME_MODEL="$NODE_CACHE/Qwen3-14B-40c06982"
VERIFIER_HOME="$NODE_CACHE/verifier-home"
VERIFIER_UV_CACHE="$NODE_CACHE/verifier-uv"
ORCHESTRATOR_ENV="{\"HOME\":\"$VERIFIER_HOME\",\"UV_CACHE_DIR\":\"$VERIFIER_UV_CACHE\"}"
export VLLM_CACHE_ROOT="$NODE_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$NODE_CACHE/torchinductor"
export TRITON_CACHE_DIR="$NODE_CACHE/triton"
mkdir -p "$RUNTIME_MODEL" "$VERIFIER_HOME" "$VERIFIER_UV_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
df -h "$NODE_CACHE"
test "$(df -Pk "$NODE_CACHE" | awk 'NR == 2 {print $4}')" -ge 83886080
if ! test -s "$RUNTIME_MODEL/.stage-complete"; then
    cp -a "$MODEL_SOURCE/." "$RUNTIME_MODEL/"
    printf 'complete\n' > "$RUNTIME_MODEL/.stage-complete"
fi
test -s "$RUNTIME_MODEL/model.safetensors.index.json"
PRIME_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}
if test "$PRIME_VISIBLE_DEVICES" = 0,1,2,3,4,5,6,7,8; then
    PRIME_VISIBLE_DEVICES=7,8,0,1,2,3,4,5,6
fi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-unset}"
printf 'PRIME_VISIBLE_DEVICES=%s\n' "$PRIME_VISIBLE_DEVICES"
env CUDA_VISIBLE_DEVICES="$PRIME_VISIBLE_DEVICES" uv run --no-sync rl @ "$EXPERIMENT_ROOT/config/main.toml" @ "$EXPERIMENT_ROOT/config/smoke.toml" --model.name "$RUNTIME_MODEL" --run.name "qwen3-14b-grpo-smoke-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}" --orchestrator.env-vars "$ORCHESTRATOR_ENV"
SMOKE_DIR="$RL_INFRA/outputs/dapo-qwen3-14b-grpo/qwen3-14b-grpo-smoke-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}"
test -s "$SMOKE_DIR/checkpoints/step_2/trainer/.metadata"
test -s "$SMOKE_DIR/checkpoints/step_2/orchestrator/progress.pt"
SMOKE_LINK=/mnt/xfs/home/mohamadzbib/projects/models/qwen3-14b-grpo-smoke-$SLURM_JOB_ID
if ! test -e "$SMOKE_LINK" && ! test -L "$SMOKE_LINK"; then
    ln -s "$SMOKE_DIR" "$SMOKE_LINK"
fi
