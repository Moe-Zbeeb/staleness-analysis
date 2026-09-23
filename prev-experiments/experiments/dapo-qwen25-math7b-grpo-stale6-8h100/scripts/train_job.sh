set -euo pipefail
EXPERIMENT_ROOT=/mnt/nfs/home/mohamadzbib/projects/rl-infra/experiments/dapo-qwen25-math7b-grpo-stale6-8h100
RL_INFRA=/mnt/nfs/home/mohamadzbib/projects/rl-infra
RUN_NAME=qwen25-math7b-grpo-seed42-stale6-8h100
ATTEMPT_ID="job-$SLURM_JOB_ID-restart-${SLURM_RESTART_COUNT:-0}"
RUN_DIR="$RL_INFRA/outputs/dapo-qwen25-math7b-grpo-stale6-8h100/$RUN_NAME"
OBSERVER_DIR="$RL_INFRA/outputs/dapo-qwen25-math7b-grpo-stale6-8h100-observer/$RUN_NAME"
RECOVERY_DIR="$RL_INFRA/outputs/dapo-qwen25-math7b-grpo-stale6-8h100-recovery/$RUN_NAME"
OBSERVER_ARCHIVE="$RL_INFRA/artifacts/opik-2.2.61-py312.tar"
CREDENTIALS=/mnt/nfs/home/mohamadzbib/.config/rl-infra/opik.env
STOP_FILE="$OBSERVER_DIR/opik.stop"
READY_FILE="$OBSERVER_DIR/opik.ready"
STATE_FILE="$OBSERVER_DIR/opik-state.json"
COMPLETE_FILE="$OBSERVER_DIR/training.complete"
TRAIN_FINISHED_FILE="$OBSERVER_DIR/training-finished.sha256"
RUN_MANIFEST_FILE="$OBSERVER_DIR/run-manifest.sha256"
METRICS_FILE="$RUN_DIR/metrics.jsonl"
MODEL_LINK=/mnt/xfs/home/mohamadzbib/projects/models/qwen25-math7b-grpo-seed42-stale6-8h100
MODEL_SOURCE=/mnt/nfs/home/mohamadzbib/projects/models/Qwen2.5-Math-7B-b101308f
ACTIVE_PID=
stop_active() {
    if test -n "$ACTIVE_PID" && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$ACTIVE_PID" 2>/dev/null; then
            kill -KILL "$ACTIVE_PID" 2>/dev/null || true
        fi
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    ACTIVE_PID=
}
bootstrap_requeue() {
    stop_active
    if ! scontrol requeue "$SLURM_JOB_ID"; then
        exit 1
    fi
    exit 0
}
bootstrap_terminate() {
    stop_active
    exit 143
}
run_guarded() {
    "$@" &
    ACTIVE_PID=$!
    set +e
    wait "$ACTIVE_PID"
    COMMAND_STATUS=$?
    set -e
    ACTIVE_PID=
    return "$COMMAND_STATUS"
}
trap bootstrap_requeue USR1
trap bootstrap_terminate TERM INT
source "$RL_INFRA/env.sh" prime-rl
uv run --no-sync python "$EXPERIMENT_ROOT/scripts/staleness_guard.py" gate --experiment-root "$EXPERIMENT_ROOT"
unset OPIK_API_KEY OPIK_URL_OVERRIDE OPIK_WORKSPACE OPIK_PROJECT_NAME
export PYTHONPATH="$EXPERIMENT_ROOT/python"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
test "$(git -C "$RL_INFRA/repos/prime-rl" rev-parse HEAD)" = ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
test -z "$(git -C "$RL_INFRA/repos/prime-rl" status --porcelain)"
verify_credentials() (
    test -s "$CREDENTIALS"
    test "$(stat -c %u "$CREDENTIALS")" = "$(id -u)"
    test "$(stat -c %a "$CREDENTIALS")" = 600
    if command -v getfacl >/dev/null 2>&1; then
        ! getfacl -cp "$CREDENTIALS" | grep -Eq '^(user|group):[^:]+:'
    fi
    unset OPIK_API_KEY OPIK_URL_OVERRIDE OPIK_WORKSPACE OPIK_PROJECT_NAME
    set -a
    source "$CREDENTIALS"
    set +a
    : "${OPIK_API_KEY:?}"
    test "$OPIK_URL_OVERRIDE" = https://www.comet.com/opik/api
    test "$OPIK_WORKSPACE" = mohamad-zbib-0046
    export OPIK_PROJECT_NAME=qwen25-math7b-grpo-stale6-8h100
)
run_guarded uv run --no-sync python "$EXPERIMENT_ROOT/scripts/validate_experiment.py" --experiment-root "$EXPERIMENT_ROOT"
MANIFEST_SHA256=$(sha256sum "$EXPERIMENT_ROOT/data/manifest.json" | awk '{print $1}')
if test -f "$COMPLETE_FILE"; then
    test -s "$RUN_MANIFEST_FILE"
    test -s "$TRAIN_FINISHED_FILE"
    test "$(cat "$RUN_MANIFEST_FILE")" = "$MANIFEST_SHA256"
    test "$(cat "$TRAIN_FINISHED_FILE")" = "$MANIFEST_SHA256"
    test "$(cat "$COMPLETE_FILE")" = "$MANIFEST_SHA256"
    test -s "$RUN_DIR/checkpoints/step_1000/trainer/.metadata"
    test -s "$RUN_DIR/checkpoints/step_1000/orchestrator/progress.pt"
    exit 0
fi
verify_credentials
NODE_CACHE=${SLURM_TMPDIR:-/tmp/prime-rl-$SLURM_JOB_ID}
RUNTIME_MODEL="$NODE_CACHE/Qwen2.5-Math-7B-b101308f"
OBSERVER_ENV="$NODE_CACHE/opik-env"
VERIFIER_HOME="$NODE_CACHE/verifier-home"
VERIFIER_UV_CACHE="$NODE_CACHE/verifier-uv"
ORCHESTRATOR_ENV="{\"HOME\":\"$VERIFIER_HOME\",\"UV_CACHE_DIR\":\"$VERIFIER_UV_CACHE\"}"
export VLLM_CACHE_ROOT="$NODE_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$NODE_CACHE/torchinductor"
export TRITON_CACHE_DIR="$NODE_CACHE/triton"
mkdir -p "$RUNTIME_MODEL" "$VERIFIER_HOME" "$VERIFIER_UV_CACHE" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
df -h "$NODE_CACHE"
test "$(df -Pk "$NODE_CACHE" | awk 'NR == 2 {print $4}')" -ge 41943040
if ! test -s "$OBSERVER_ENV/.stage-complete"; then
    test -s "$OBSERVER_ARCHIVE"
    tar -C "$NODE_CACHE" -xf "$OBSERVER_ARCHIVE"
    printf 'complete\n' > "$OBSERVER_ENV/.stage-complete"
fi
"$OBSERVER_ENV/bin/python" -c 'import inspect, opik; assert opik.__version__ == "2.2.61"; assert "feedback_scores" in inspect.signature(opik.Opik.trace).parameters'
mkdir -p "$OBSERVER_DIR" "$RECOVERY_DIR"
if test -s "$RUN_MANIFEST_FILE"; then
    test "$(cat "$RUN_MANIFEST_FILE")" = "$MANIFEST_SHA256"
else
    test ! -e "$RUN_DIR"
    test ! -e "$STATE_FILE"
    printf '%s\n' "$MANIFEST_SHA256" > "$RUN_MANIFEST_FILE.job-$SLURM_JOB_ID.tmp"
    mv "$RUN_MANIFEST_FILE.job-$SLURM_JOB_ID.tmp" "$RUN_MANIFEST_FILE"
fi
TRAINING_FINISHED=false
if test -f "$TRAIN_FINISHED_FILE"; then
    test "$(cat "$TRAIN_FINISHED_FILE")" = "$MANIFEST_SHA256"
    test -s "$RUN_DIR/checkpoints/step_1000/trainer/.metadata"
    test -s "$RUN_DIR/checkpoints/step_1000/orchestrator/progress.pt"
    TRAINING_FINISHED=true
fi
if test "$TRAINING_FINISHED" = false; then
    mkdir -p "$(dirname "$MODEL_LINK")"
    if test -L "$MODEL_LINK"; then
        test "$(readlink "$MODEL_LINK")" = "$RUN_DIR"
    elif test -e "$MODEL_LINK"; then
        exit 1
    else
        ln -s "$RUN_DIR" "$MODEL_LINK"
    fi
    timeout -k 10 30 nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,uuid,memory.total,temperature.gpu,power.draw --format=csv
    run_guarded timeout -k 15 240 uv run --no-sync torchrun --standalone --nproc-per-node=8 "$EXPERIMENT_ROOT/scripts/health_probe.py"
    if ! test -s "$RUNTIME_MODEL/.stage-complete"; then
        run_guarded cp -a "$MODEL_SOURCE/." "$RUNTIME_MODEL/"
        printf 'complete\n' > "$RUNTIME_MODEL/.stage-complete"
    fi
    test -s "$RUNTIME_MODEL/model.safetensors.index.json"
fi
rm -f "$STOP_FILE" "$READY_FILE"
ARGS=(uv run --no-sync rl @ "$EXPERIMENT_ROOT/config/main.toml" --model.name "$RUNTIME_MODEL" --orchestrator.env-vars "$ORCHESTRATOR_ENV")
PRIME_VISIBLE_DEVICES=$(uv run --no-sync python "$EXPERIMENT_ROOT/scripts/staleness_guard.py" devices --experiment-root "$EXPERIMENT_ROOT")
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-unset}"
printf 'PRIME_VISIBLE_DEVICES=%s\n' "$PRIME_VISIBLE_DEVICES"
RESUME_STEP=
if test "$TRAINING_FINISHED" = false && test -d "$RUN_DIR/checkpoints"; then
    CHECKPOINT_RECOVERY="$RECOVERY_DIR/checkpoints/job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}"
    for CHECKPOINT in "$RUN_DIR"/checkpoints/step_*; do
        test -d "$CHECKPOINT" || continue
        STEP=${CHECKPOINT##*/step_}
        case "$STEP" in
            *[!0-9]*|'') continue ;;
        esac
        if test -s "$CHECKPOINT/trainer/.metadata" && test -s "$CHECKPOINT/orchestrator/progress.pt" && test "$STEP" -lt 1000; then
            if test -z "$RESUME_STEP" || test "$STEP" -gt "$RESUME_STEP"; then
                RESUME_STEP=$STEP
            fi
        else
            mkdir -p "$CHECKPOINT_RECOVERY"
            mv "$CHECKPOINT" "$CHECKPOINT_RECOVERY/"
        fi
    done
fi
if test "$TRAINING_FINISHED" = true; then
    :
elif test -n "$RESUME_STEP"; then
    ARGS+=(--resume.step "$RESUME_STEP")
elif test -d "$RUN_DIR"; then
    RUN_RECOVERY="$RECOVERY_DIR/run-job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}-pid-$$"
    mv "$RUN_DIR" "$RUN_RECOVERY"
    if test -f "$STATE_FILE"; then
        mv "$STATE_FILE" "$RUN_RECOVERY/opik-state.json"
    fi
elif test -f "$STATE_FILE"; then
    mv "$STATE_FILE" "$RECOVERY_DIR/orphan-opik-state-job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}-pid-$$.json"
fi
verify_credentials
unset OPIK_API_KEY OPIK_URL_OVERRIDE OPIK_WORKSPACE OPIK_PROJECT_NAME
set -a
source "$CREDENTIALS"
set +a
: "${OPIK_API_KEY:?}"
test "$OPIK_URL_OVERRIDE" = https://www.comet.com/opik/api
test "$OPIK_WORKSPACE" = mohamad-zbib-0046
export OPIK_PROJECT_NAME=qwen25-math7b-grpo-stale6-8h100
"$OBSERVER_ENV/bin/python" "$EXPERIMENT_ROOT/scripts/opik_bridge.py" --metrics "$METRICS_FILE" --state "$STATE_FILE" --stop-file "$STOP_FILE" --ready-file "$READY_FILE" --run-name "$RUN_NAME" --attempt-id "$ATTEMPT_ID" --project "$OPIK_PROJECT_NAME" --workspace "$OPIK_WORKSPACE" --prime-commit ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1 &
BRIDGE_PID=$!
unset OPIK_API_KEY OPIK_URL_OVERRIDE OPIK_WORKSPACE OPIK_PROJECT_NAME
TRAIN_PID=
BRIDGE_STATUS=0
wait_for_exit() {
    PID=$1
    LIMIT=$2
    for _ in $(seq 1 "$LIMIT"); do
        if ! kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}
stop_training() {
    if test -n "$TRAIN_PID" && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID"
        if ! wait_for_exit "$TRAIN_PID" 120; then
            kill -KILL "$TRAIN_PID" 2>/dev/null || true
        fi
        wait "$TRAIN_PID" || true
    fi
    TRAIN_PID=
}
stop_bridge() {
    touch "$STOP_FILE"
    if kill -0 "$BRIDGE_PID" 2>/dev/null && ! wait_for_exit "$BRIDGE_PID" 45; then
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
        if ! wait_for_exit "$BRIDGE_PID" 20; then
            kill -KILL "$BRIDGE_PID" 2>/dev/null || true
        fi
    fi
    set +e
    wait "$BRIDGE_PID"
    BRIDGE_STATUS=$?
    set -e
}
finish() {
    stop_training
    stop_bridge
}
requeue() {
    finish
    if ! scontrol requeue "$SLURM_JOB_ID"; then
        exit 1
    fi
    exit 0
}
terminate() {
    finish
    exit 143
}
trap requeue USR1
trap terminate TERM INT
for _ in $(seq 1 90); do
    if test -s "$READY_FILE"; then
        break
    fi
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        set +e
        wait "$BRIDGE_PID"
        BRIDGE_STATUS=$?
        set -e
        if test "$BRIDGE_STATUS" -eq 0; then
            BRIDGE_STATUS=1
        fi
        exit "$BRIDGE_STATUS"
    fi
    sleep 1
done
if ! test -s "$READY_FILE"; then
    stop_bridge
    if test "$BRIDGE_STATUS" -eq 0; then
        BRIDGE_STATUS=1
    fi
    exit "$BRIDGE_STATUS"
fi
if test "$TRAINING_FINISHED" = true; then
    stop_bridge
    if test "$BRIDGE_STATUS" -ne 0; then
        exit "$BRIDGE_STATUS"
    fi
    printf '%s\n' "$MANIFEST_SHA256" > "$COMPLETE_FILE.job-$SLURM_JOB_ID.tmp"
    mv "$COMPLETE_FILE.job-$SLURM_JOB_ID.tmp" "$COMPLETE_FILE"
    exit 0
fi
env CUDA_VISIBLE_DEVICES="$PRIME_VISIBLE_DEVICES" "${ARGS[@]}" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        set +e
        wait "$BRIDGE_PID"
        BRIDGE_STATUS=$?
        set -e
        if test "$BRIDGE_STATUS" -eq 0; then
            BRIDGE_STATUS=1
        fi
        stop_training
        exit "$BRIDGE_STATUS"
    fi
    sleep 5
done
set +e
wait "$TRAIN_PID"
STATUS=$?
set -e
TRAIN_PID=
if test "$STATUS" -eq 0; then
    test -s "$RUN_DIR/checkpoints/step_1000/trainer/.metadata"
    test -s "$RUN_DIR/checkpoints/step_1000/orchestrator/progress.pt"
    uv run --no-sync python "$EXPERIMENT_ROOT/scripts/staleness_guard.py" audit --experiment-root "$EXPERIMENT_ROOT" --run-dir "$RUN_DIR" --steps 1000
    printf '%s\n' "$MANIFEST_SHA256" > "$TRAIN_FINISHED_FILE.job-$SLURM_JOB_ID.tmp"
    mv "$TRAIN_FINISHED_FILE.job-$SLURM_JOB_ID.tmp" "$TRAIN_FINISHED_FILE"
fi
stop_bridge
if test "$STATUS" -eq 0 && test "$BRIDGE_STATUS" -eq 0; then
    printf '%s\n' "$MANIFEST_SHA256" > "$COMPLETE_FILE.job-$SLURM_JOB_ID.tmp"
    mv "$COMPLETE_FILE.job-$SLURM_JOB_ID.tmp" "$COMPLETE_FILE"
    exit 0
fi
if test "$STATUS" -ne 0; then
    exit "$STATUS"
fi
exit "$BRIDGE_STATUS"
