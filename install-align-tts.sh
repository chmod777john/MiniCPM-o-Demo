#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-align-tts}"
PYTHON="${PYTHON:-/user/houyueran/miniconda/envs/cpmo_35A3/bin/python}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
LOCK_FILE="${PROJECT_DIR}/requirements-align-tts-py311-cu128.lock"

# Do not inherit machine-level indexes such as an unavailable NVIDIA NGC
# mirror. Callers can still provide HTTP(S)_PROXY or ALL_PROXY explicitly.
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"

if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing virtualenv Python: ${PYTHON_BIN}" >&2
    exit 1
fi

# Install the build/runtime foundation first so CUDA extensions can inspect
# the final Torch ABI while their metadata or wheels are prepared.
"${PYTHON_BIN}" -m pip install \
    "pip==26.1.1" \
    "setuptools==80.10.2" \
    "wheel==0.46.3"
"${PYTHON_BIN}" -m pip install \
    --extra-index-url "${TORCH_INDEX}" \
    "torch==2.8.0+cu128" \
    "torchvision==0.23.0+cu128" \
    "torchaudio==2.8.0+cu128" \
    "triton==3.4.0"

MAX_JOBS="${MAX_JOBS}" "${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    -r "${LOCK_FILE}"

# minicpmo-utils pins librosa==0.9.0 in its package metadata. Keep the Demo
# service's librosa==0.11.0 by installing the utility package without deps.
"${PYTHON_BIN}" -m pip install --no-deps "minicpmo-utils==1.0.6"

"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.8.0+cu128",
    "transformers": "5.12.0",
    "numpy": "1.26.4",
    "librosa": "0.11.0",
    "minicpmo-utils": "1.0.6",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"{package}: expected {wanted}, found {actual}")
    print(f"{package}=={actual}")
PY

echo "TTS-aligned environment ready: ${VENV_DIR}"
