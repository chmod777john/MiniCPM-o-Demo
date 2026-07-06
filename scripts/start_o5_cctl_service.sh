#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
PT_PATH="${PT_PATH:-}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-22510}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
WORKER_PORT="${WORKER_PORT:-22410}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"
GPU_ID="${GPU_ID:-0}"

WORKER_ID="${WORKER_ID:-o5-cctl-worker}"
WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-${GPU_ID}}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/run-logs/o5_cctl}"
ENABLE_FRP="${ENABLE_FRP:-1}"
FRPC_BIN="${FRPC_BIN:-frpc}"
FRPC_CONFIG="${FRPC_CONFIG:-}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${VENV_DIR}/bin/python"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
WORKER_ENDPOINT="${WORKER_HOST}:${WORKER_PORT}"
GATEWAY_REGISTRY_URL="http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/internal/workers/${WORKER_ID}"

if [ ! -x "${PYTHON}" ]; then
    echo "[start] missing python: ${PYTHON}" >&2
    exit 1
fi
if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then
    echo "[start] missing model dir: ${MODEL_PATH}" >&2
    echo "[start] set MODEL_PATH=/path/to/model-code-and-tokenizer" >&2
    exit 1
fi
if [ -z "${PT_PATH}" ] || [ ! -f "${PT_PATH}" ]; then
    echo "[start] missing pt file: ${PT_PATH}" >&2
    echo "[start] set PT_PATH=/path/to/checkpoint.pt" >&2
    exit 1
fi

backend_pid=""
worker_pid=""
gateway_pid=""
frpc_pid=""

cleanup() {
    echo "[start] cleanup"
    [ -n "${frpc_pid}" ] && kill "${frpc_pid}" 2>/dev/null || true
    [ -n "${worker_pid}" ] && kill "${worker_pid}" 2>/dev/null || true
    [ -n "${backend_pid}" ] && kill "${backend_pid}" 2>/dev/null || true
    [ -n "${gateway_pid}" ] && kill "${gateway_pid}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_http() {
    local url="$1"
    local timeout_s="$2"
    local name="$3"
    local elapsed=0
    until curl -sf "${url}" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "${elapsed}" -ge "${timeout_s}" ]; then
            echo "[start] timeout waiting for ${name}: ${url}" >&2
            return 1
        fi
    done
    echo "[start] ${name} ready after ${elapsed}s"
}

echo "[start] project=${PROJECT_DIR}"
echo "[start] model=${MODEL_PATH}"
echo "[start] pt=${PT_PATH}"
echo "[start] gateway=https://${GATEWAY_HOST}:${GATEWAY_PORT} internal=:${GATEWAY_INTERNAL_PORT}"
echo "[start] backend=${BACKEND_URL} worker=${WORKER_ENDPOINT}"

"${PYTHON}" gateway.py \
    --host "${GATEWAY_HOST}" \
    --port "${GATEWAY_PORT}" \
    --internal-port "${GATEWAY_INTERNAL_PORT}" \
    --https \
    --ssl-certfile certs/cert.pem \
    --ssl-keyfile certs/key.pem \
    > "${LOG_DIR}/gateway.log" 2>&1 &
gateway_pid=$!
wait_http "http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/health" 120 "gateway-internal"

"${PYTHON}" -m py_backend.server \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" \
    --model-path "${MODEL_PATH}" \
    --pt-path "${PT_PATH}" \
    --gpu-id "${GPU_ID}" \
    > "${LOG_DIR}/backend.log" 2>&1 &
backend_pid=$!
wait_http "${BACKEND_URL}/health" 900 "backend"

"${PYTHON}" worker.py \
    --host "${WORKER_HOST}" \
    --port "${WORKER_PORT}" \
    --gpu-id "${GPU_ID}" \
    --backend-server-url "${BACKEND_URL}" \
    > "${LOG_DIR}/worker.log" 2>&1 &
worker_pid=$!
wait_http "http://${WORKER_ENDPOINT}/health" 120 "worker"

payload="{\"endpoint\":\"${WORKER_ENDPOINT}\",\"gpu_group\":\"${WORKER_GPU_GROUP}\",\"labels\":{\"model\":\"o5\",\"runtime\":\"cctl\"}}"
curl -sf -X PUT \
    -H "content-type: application/json" \
    --data "${payload}" \
    "${GATEWAY_REGISTRY_URL}" >/dev/null
echo "[start] registered ${WORKER_ID} endpoint=${WORKER_ENDPOINT}"

curl -sk "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null
echo "[start] service ready"

if [ "${ENABLE_FRP}" = "1" ]; then
    if [ -z "${FRPC_CONFIG}" ]; then
        echo "[start] ENABLE_FRP=1 requires FRPC_CONFIG=/path/to/frpc.toml" >&2
        exit 1
    fi
    if [ ! -x "${FRPC_BIN}" ]; then
        echo "[start] missing frpc: ${FRPC_BIN}" >&2
        exit 1
    fi
    if [ ! -f "${FRPC_CONFIG}" ]; then
        echo "[start] missing frpc config: ${FRPC_CONFIG}" >&2
        exit 1
    fi
    "${FRPC_BIN}" -c "${FRPC_CONFIG}" > "${LOG_DIR}/frpc.log" 2>&1 &
    frpc_pid=$!
    sleep 3
    if ! kill -0 "${frpc_pid}" 2>/dev/null; then
        echo "[start] frpc exited during startup" >&2
        tail -n 80 "${LOG_DIR}/frpc.log" >&2 || true
        exit 1
    fi
    echo "[start] frpc ready: ${FRPC_CONFIG}"
fi

while true; do
    if [ -n "${frpc_pid}" ] && ! kill -0 "${frpc_pid}" 2>/dev/null; then
        echo "[start] frpc exited" >&2
        exit 1
    fi
    if ! kill -0 "${gateway_pid}" 2>/dev/null; then
        echo "[start] gateway exited" >&2
        exit 1
    fi
    if ! kill -0 "${backend_pid}" 2>/dev/null; then
        echo "[start] backend exited" >&2
        exit 1
    fi
    if ! kill -0 "${worker_pid}" 2>/dev/null; then
        echo "[start] worker exited" >&2
        exit 1
    fi
    sleep 5
done
