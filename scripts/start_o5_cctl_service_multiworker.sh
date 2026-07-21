#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
PT_PATH="${PT_PATH:-}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_BASE_PORT="${BACKEND_BASE_PORT:-22510}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-22410}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BACKEND_START_STAGGER_SECONDS="${BACKEND_START_STAGGER_SECONDS:-90}"

WORKER_ID_PREFIX="${WORKER_ID_PREFIX:-o5-cctl-worker}"
WORKER_GPU_GROUP_PREFIX="${WORKER_GPU_GROUP_PREFIX:-cctl-a100}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/run-logs/o5_cctl_multiworker}"
ENABLE_FRP="${ENABLE_FRP:-1}"
FRPC_BIN="${FRPC_BIN:-frpc}"
FRPC_CONFIG="${FRPC_CONFIG:-}"
FRPC_RETRY_SECONDS="${FRPC_RETRY_SECONDS:-10}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${PYTHON}" ]; then
    echo "[start] missing python: ${PYTHON}" >&2
    exit 1
fi
if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then
    echo "[start] missing model dir: ${MODEL_PATH}" >&2
    exit 1
fi
if [ -z "${PT_PATH}" ] || [ ! -f "${PT_PATH}" ]; then
    echo "[start] missing pt file: ${PT_PATH}" >&2
    exit 1
fi

pids=()
frpc_loop_pid=""

cleanup() {
    echo "[start] cleanup"
    [ -n "${frpc_loop_pid}" ] && kill "${frpc_loop_pid}" 2>/dev/null || true
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
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

start_frpc_retry_loop() {
    if [ "${ENABLE_FRP}" != "1" ]; then
        echo "[start] frp disabled"
        return 0
    fi
    if [ -z "${FRPC_CONFIG}" ] || [ ! -f "${FRPC_CONFIG}" ]; then
        echo "[start] missing frpc config: ${FRPC_CONFIG}" >&2
        exit 1
    fi
    if [ ! -x "${FRPC_BIN}" ]; then
        echo "[start] missing frpc: ${FRPC_BIN}" >&2
        exit 1
    fi

    (
        attempt=0
        while true; do
            attempt=$((attempt + 1))
            echo "[frpc-loop] attempt ${attempt}: ${FRPC_CONFIG}"
            "${FRPC_BIN}" -c "${FRPC_CONFIG}" >> "${LOG_DIR}/frpc.log" 2>&1
            code=$?
            echo "[frpc-loop] frpc exited code=${code}; retry in ${FRPC_RETRY_SECONDS}s"
            sleep "${FRPC_RETRY_SECONDS}"
        done
    ) > "${LOG_DIR}/frpc-loop.log" 2>&1 &
    frpc_loop_pid=$!
    echo "[start] frpc retry loop pid=${frpc_loop_pid}"
}

echo "[start] project=${PROJECT_DIR}"
echo "[start] model=${MODEL_PATH}"
echo "[start] pt=${PT_PATH}"
echo "[start] workers=${NUM_WORKERS} backend_base=${BACKEND_BASE_PORT} worker_base=${WORKER_BASE_PORT}"
echo "[start] backend_start_stagger_seconds=${BACKEND_START_STAGGER_SECONDS}"
echo "[start] gateway=https://${GATEWAY_HOST}:${GATEWAY_PORT} internal=:${GATEWAY_INTERNAL_PORT}"

"${PYTHON}" gateway.py \
    --host "${GATEWAY_HOST}" \
    --port "${GATEWAY_PORT}" \
    --internal-port "${GATEWAY_INTERNAL_PORT}" \
    --https \
    --ssl-certfile certs/cert.pem \
    --ssl-keyfile certs/key.pem \
    > "${LOG_DIR}/gateway.log" 2>&1 &
pids+=("$!")
wait_http "http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/health" 120 "gateway-internal"

backend_ports=()
backend_urls=()
worker_ports=()
worker_endpoints=()

for ((i=0; i<NUM_WORKERS; i++)); do
    backend_port=$((BACKEND_BASE_PORT + i))
    worker_port=$((WORKER_BASE_PORT + i))
    gpu_id="${i}"
    backend_url="http://${BACKEND_HOST}:${backend_port}"
    worker_endpoint="${WORKER_HOST}:${worker_port}"

    backend_ports+=("${backend_port}")
    backend_urls+=("${backend_url}")
    worker_ports+=("${worker_port}")
    worker_endpoints+=("${worker_endpoint}")

    echo "[start] backend ${i}: physical_gpu=${gpu_id} visible_cuda=0 url=${backend_url}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" -m py_backend.server \
        --host "${BACKEND_HOST}" \
        --port "${backend_port}" \
        --model-path "${MODEL_PATH}" \
        --pt-path "${PT_PATH}" \
        --gpu-id 0 \
        > "${LOG_DIR}/backend_${i}.log" 2>&1 &
    pids+=("$!")
    if [ "${BACKEND_START_STAGGER_SECONDS}" -gt 0 ] && [ "$((i + 1))" -lt "${NUM_WORKERS}" ]; then
        echo "[start] waiting ${BACKEND_START_STAGGER_SECONDS}s before launching next backend"
        sleep "${BACKEND_START_STAGGER_SECONDS}"
    fi
done

for ((i=0; i<NUM_WORKERS; i++)); do
    wait_http "${backend_urls[$i]}/health" 1200 "backend-${i}"
done

for ((i=0; i<NUM_WORKERS; i++)); do
    gpu_id="${i}"
    backend_url="${backend_urls[$i]}"
    worker_endpoint="${worker_endpoints[$i]}"
    worker_id="${WORKER_ID_PREFIX}-${i}"
    gpu_group="${WORKER_GPU_GROUP_PREFIX}-${gpu_id}"

    echo "[start] worker ${i}: endpoint=${worker_endpoint} backend=${backend_url}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" worker.py \
        --host "${WORKER_HOST}" \
        --port "${worker_ports[$i]}" \
        --gpu-id 0 \
        --backend-server-url "${backend_url}" \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &
    pids+=("$!")
done

for ((i=0; i<NUM_WORKERS; i++)); do
    wait_http "http://${worker_endpoints[$i]}/health" 120 "worker-${i}"
done

for ((i=0; i<NUM_WORKERS; i++)); do
    gpu_id="${i}"
    worker_endpoint="${worker_endpoints[$i]}"
    worker_id="${WORKER_ID_PREFIX}-${i}"
    gpu_group="${WORKER_GPU_GROUP_PREFIX}-${gpu_id}"

    payload="{\"endpoint\":\"${worker_endpoint}\",\"gpu_group\":\"${gpu_group}\",\"labels\":{\"model\":\"o5\",\"runtime\":\"cctl\",\"worker_index\":\"${i}\"}}"
    curl -sf -X PUT \
        -H "content-type: application/json" \
        --data "${payload}" \
        "http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/internal/workers/${worker_id}" >/dev/null
    echo "[start] registered ${worker_id} endpoint=${worker_endpoint}"
done

curl -sk "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null
echo "[start] service ready with ${NUM_WORKERS} workers"

start_frpc_retry_loop

while true; do
    for pid in "${pids[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[start] child exited: ${pid}" >&2
            exit 1
        fi
    done
    if [ -n "${frpc_loop_pid}" ] && ! kill -0 "${frpc_loop_pid}" 2>/dev/null; then
        echo "[start] frpc loop exited" >&2
        exit 1
    fi
    sleep 5
done
