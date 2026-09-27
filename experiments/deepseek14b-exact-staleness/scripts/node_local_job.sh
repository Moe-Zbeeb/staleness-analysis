#!/bin/bash
set -euo pipefail
LOCAL_STUDY_WORKSPACE=${1:?node-local prepared workspace required}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 "$LOCAL_STUDY_WORKSPACE/node_local_run.py" run --control "$LOCAL_STUDY_WORKSPACE"
