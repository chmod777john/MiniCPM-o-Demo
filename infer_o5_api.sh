#!/usr/bin/env bash
# 使用仓库固定的可复用环境运行 TrainingData→O5 API 推理样例。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNAL_SETTINGS="${O5_API_INTERNAL_SETTINGS_PATH:-${PROJECT_DIR}/configs/fc_deployment/o5_api_internal_settings.json}"

if [ ! -f "${INTERNAL_SETTINGS}" ]; then
    echo "[o5-api-inference] missing internal settings: ${INTERNAL_SETTINGS}" >&2
    exit 2
fi

RUNTIME_PYTHON="$(
    python3 - "${INTERNAL_SETTINGS}" <<'PY'
import json
import sys
from pathlib import Path

settings_path = Path(sys.argv[1])
settings = json.loads(settings_path.read_text(encoding="utf-8"))
print(settings["runtime_python"])
PY
)"

if [ ! -x "${RUNTIME_PYTHON}" ]; then
    echo "[o5-api-inference] reusable runtime python is unavailable: ${RUNTIME_PYTHON}" >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${RUNTIME_PYTHON}" \
    "${PROJECT_DIR}/examples/fc_duplex/infer_from_training_data.py" \
    "$@"
