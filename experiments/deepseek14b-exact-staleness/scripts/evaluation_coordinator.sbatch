#!/bin/bash
#SBATCH --job-name=checkpoint-eval-coordinator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
if [[ -n "${EVAL_CLUSTER_ENV:-}" ]]; then
    source "$EVAL_CLUSTER_ENV"
fi
: "${EVAL_ROOT:?Set the persistent evaluator pool directory}"
: "${EVAL_SLURM_CONFIG:?Set the worker submission configuration}"
: "${EVAL_SOURCE:?Set the immutable release src directory}"
: "${EVAL_PYTHON:?Set the pinned evaluator runtime Python}"
: "${EVAL_UV:?Set the uv executable path}"
export PYTHONPATH="$EVAL_SOURCE"
export PYTHONUNBUFFERED=1
export UV_OFFLINE=true
child_pid=
requeue_coordinator() {
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    timeout --kill-after=5s 30s scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_coordinator USR1
"$EVAL_UV" run --no-project "$EVAL_PYTHON" -m deepseek_study.evaluation serve "$EVAL_ROOT" \
    --slurm-config "$EVAL_SLURM_CONFIG" &
child_pid=$!
wait "$child_pid"
