#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
PT_PATH="${PT_PATH:-}"
BACKBONE_DIR="${BACKBONE_DIR:-${O5_BACKBONE_DIR:-}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-22510}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
WORKER_PORT="${WORKER_PORT:-22410}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"
WORKER_ID="${WORKER_ID:-o5-tp2-cctl-worker}"
WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-tp2}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/run-logs/o5_tp2_cctl}"
O5_TOKEN_TRACE_DIR="${O5_TOKEN_TRACE_DIR:-${PROJECT_DIR}/data/sessions}"
ENABLE_FRP="${ENABLE_FRP:-0}"
FRPC_BIN="${FRPC_BIN:-frpc}"
FRPC_CONFIG="${FRPC_CONFIG:-}"
GATEWAY_HTTPS="${GATEWAY_HTTPS:-1}"
O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
O5_SPMD_HEARTBEAT_INTERVAL="${O5_SPMD_HEARTBEAT_INTERVAL:-30}"
FC_MODEL_FAMILY="${FC_MODEL_FAMILY:-o5}"
CHECKPOINT_PROFILE_ID="${CHECKPOINT_PROFILE_ID:-unprofiled}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export O5_BACKBONE_DIR="${BACKBONE_DIR}"
export O5_LLM_CACHE
export O5_SPMD_HEARTBEAT_INTERVAL
export O5_TOKEN_TRACE_DIR
export TORCHRUN="${VENV_DIR}/bin/torchrun"

PYTHON="${VENV_DIR}/bin/python"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
WORKER_ENDPOINT="${WORKER_HOST}:${WORKER_PORT}"
GATEWAY_REGISTRY_URL="http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/internal/workers/${WORKER_ID}"

if [ ! -x "${PYTHON}" ]; then echo "[tp2-start] missing python: ${PYTHON}" >&2; exit 1; fi
if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then echo "[tp2-start] missing MODEL_PATH=${MODEL_PATH}" >&2; exit 1; fi
if [ -z "${PT_PATH}" ] || [ ! -f "${PT_PATH}" ]; then echo "[tp2-start] missing PT_PATH=${PT_PATH}" >&2; exit 1; fi
if [ -z "${BACKBONE_DIR}" ] || [ ! -d "${BACKBONE_DIR}" ]; then echo "[tp2-start] missing BACKBONE_DIR=${BACKBONE_DIR}" >&2; exit 1; fi

backend_pid=""; worker_pid=""; gateway_pid=""; frpc_pid=""
cleanup() {
    echo "[tp2-start] cleanup"
    [ -n "${frpc_pid}" ] && kill "${frpc_pid}" 2>/dev/null || true
    [ -n "${worker_pid}" ] && kill "${worker_pid}" 2>/dev/null || true
    [ -n "${backend_pid}" ] && kill "${backend_pid}" 2>/dev/null || true
    [ -n "${gateway_pid}" ] && kill "${gateway_pid}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_http() {
    local url="$1" timeout_s="$2" name="$3" elapsed=0
    until curl -sf "${url}" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ "${elapsed}" -ge "${timeout_s}" ]; then
            echo "[tp2-start] timeout waiting for ${name}: ${url}" >&2
            return 1
        fi
    done
    echo "[tp2-start] ${name} ready after ${elapsed}s"
}

echo "[tp2-start] project=${PROJECT_DIR}"
echo "[tp2-start] model=${MODEL_PATH}"
echo "[tp2-start] pt=${PT_PATH}"
echo "[tp2-start] token_trace_dir=${O5_TOKEN_TRACE_DIR}"
echo "[tp2-start] backbone=${BACKBONE_DIR} llm_cache=${O5_LLM_CACHE} spmd_heartbeat=${O5_SPMD_HEARTBEAT_INTERVAL}"
if [ "${GATEWAY_HTTPS}" = "1" ]; then
    gateway_scheme="https"
    gateway_args=(--https --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem)
else
    gateway_scheme="http"
    gateway_args=(--http)
fi

echo "[tp2-start] gateway=${gateway_scheme}://${GATEWAY_HOST}:${GATEWAY_PORT} backend=${BACKEND_URL} worker=${WORKER_ENDPOINT}"

"${PYTHON}" gateway.py \
    --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}" --internal-port "${GATEWAY_INTERNAL_PORT}" \
    "${gateway_args[@]}" \
    > "${LOG_DIR}/gateway.log" 2>&1 &
gateway_pid=$!
wait_http "http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/health" 120 "gateway-internal"

"${PROJECT_DIR}/core/deploy/launch_tp2.sh" \
    --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" \
    --model-path "${MODEL_PATH}" --pt-path "${PT_PATH}" \
    > "${LOG_DIR}/backend_tp2.log" 2>&1 &
backend_pid=$!
wait_http "${BACKEND_URL}/health" 1200 "backend-tp2"

"${PYTHON}" worker.py \
    --host "${WORKER_HOST}" --port "${WORKER_PORT}" \
    --gpu-id 0 --backend-server-url "${BACKEND_URL}" \
    > "${LOG_DIR}/worker.log" 2>&1 &
worker_pid=$!
wait_http "http://${WORKER_ENDPOINT}/health" 120 "worker"

payload="{\"endpoint\":\"${WORKER_ENDPOINT}\",\"gpu_group\":\"${WORKER_GPU_GROUP}\",\"labels\":{\"model\":\"${FC_MODEL_FAMILY}\",\"runtime\":\"tp2-cctl\",\"profile\":\"${CHECKPOINT_PROFILE_ID}\"}}"
curl -sf -X PUT -H "content-type: application/json" --data "${payload}" "${GATEWAY_REGISTRY_URL}" >/dev/null
echo "[tp2-start] registered ${WORKER_ID} endpoint=${WORKER_ENDPOINT}"

curl -sk "${gateway_scheme}://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null
echo "[tp2-start] service ready"

if [ "${ENABLE_FRP}" = "1" ]; then
    if [ -z "${FRPC_CONFIG}" ] || [ ! -f "${FRPC_CONFIG}" ]; then echo "[tp2-start] missing FRPC_CONFIG=${FRPC_CONFIG}" >&2; exit 1; fi
    if [ ! -x "${FRPC_BIN}" ]; then echo "[tp2-start] missing frpc: ${FRPC_BIN}" >&2; exit 1; fi
    "${FRPC_BIN}" -c "${FRPC_CONFIG}" > "${LOG_DIR}/frpc.log" 2>&1 &
    frpc_pid=$!
    sleep 3
    if ! kill -0 "${frpc_pid}" 2>/dev/null; then tail -n 80 "${LOG_DIR}/frpc.log" >&2 || true; exit 1; fi
    echo "[tp2-start] frpc ready: ${FRPC_CONFIG}"
fi

while true; do
    [ -n "${frpc_pid}" ] && ! kill -0 "${frpc_pid}" 2>/dev/null && echo "[tp2-start] frpc exited" >&2 && exit 1
    ! kill -0 "${gateway_pid}" 2>/dev/null && echo "[tp2-start] gateway exited" >&2 && exit 1
    ! kill -0 "${backend_pid}" 2>/dev/null && echo "[tp2-start] backend exited" >&2 && exit 1
    ! kill -0 "${worker_pid}" 2>/dev/null && echo "[tp2-start] worker exited" >&2 && exit 1
    sleep 5
done
