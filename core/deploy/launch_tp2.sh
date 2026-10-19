#!/usr/bin/env bash
# External HF/accel owns both visible GPUs in one process.
set -euo pipefail
: "${O5_CANONICAL_PATH:?set to the external HF source directory}"
: "${O5_ACCELERATION_PATH:?set to the external accel source directory}"
export O5_DEPLOY_MODE=${O5_DEPLOY_MODE:-tp2}
PYTHON=${PYTHON:-python}
exec "$PYTHON" -m py_backend.server "$@"
