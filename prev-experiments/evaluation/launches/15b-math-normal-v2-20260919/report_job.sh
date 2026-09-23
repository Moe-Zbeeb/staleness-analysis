#!/bin/bash
set -euo pipefail
source /mnt/nfs/home/mohamadzbib/projects/rl-infra/env.sh prime-rl
SWEEP_LAUNCH=/mnt/nfs/home/mohamadzbib/projects/rl-infra/evaluation-runs/15b-math-normal-v2-20260919/source
cd "$SWEEP_LAUNCH"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" report.py
