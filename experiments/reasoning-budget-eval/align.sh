set -euo pipefail
ROLLOUT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"${GENERATION_PYTHON:?Set GENERATION_PYTHON to the vLLM environment Python}" "$ROLLOUT_ROOT/align_tokenizers.py"
