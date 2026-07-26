#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
PT_PATH="${PT_PATH:-}"
BACKBONE_DIR="${BACKBONE_DIR:-${O5_BACKBONE_DIR:-}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_BASE_PORT="${BACKEND_BASE_PORT:-22510}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-22410}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-2}"
BACKEND_START_STAGGER_SECONDS="${BACKEND_START_STAGGER_SECONDS:-120}"
TORCHRUN_MASTER_BASE_PORT="${TORCHRUN_MASTER_BASE_PORT:-29500}"

WORKER_ID_PREFIX="${WORKER_ID_PREFIX:-o5-tp2-cctl-worker}"
WORKER_GPU_GROUP_PREFIX="${WORKER_GPU_GROUP_PREFIX:-cctl-a100-tp2}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/run-logs/o5_tp2_cctl_multiworker}"
O5_TOKEN_TRACE_DIR="${O5_TOKEN_TRACE_DIR:-${PROJECT_DIR}/data/sessions}"
ENABLE_FRP="${ENABLE_FRP:-1}"
FRPC_BIN="${FRPC_BIN:-frpc}"
FRPC_CONFIG="${FRPC_CONFIG:-}"
FRPC_RETRY_SECONDS="${FRPC_RETRY_SECONDS:-10}"
O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
O5_SPMD_HEARTBEAT_INTERVAL="${O5_SPMD_HEARTBEAT_INTERVAL:-30}"

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

if [ ! -x "${PYTHON}" ]; then echo "[tp2-mw-start] missing python: ${PYTHON}" >&2; exit 1; fi
if [ -z "${MODEL_PATH}" ] || [ ! -d "${MODEL_PATH}" ]; then echo "[tp2-mw-start] missing MODEL_PATH=${MODEL_PATH}" >&2; exit 1; fi
if [ -z "${PT_PATH}" ] || [ ! -f "${PT_PATH}" ]; then echo "[tp2-mw-start] missing PT_PATH=${PT_PATH}" >&2; exit 1; fi
if [ -z "${BACKBONE_DIR}" ] || [ ! -d "${BACKBONE_DIR}" ]; then echo "[tp2-mw-start] missing BACKBONE_DIR=${BACKBONE_DIR}" >&2; exit 1; fi
if [ "${GPUS_PER_WORKER}" -ne 2 ]; then echo "[tp2-mw-start] TP2 requires GPUS_PER_WORKER=2" >&2; exit 1; fi

pids=()
frpc_loop_pid=""

cleanup() {
    echo "[tp2-mw-start] cleanup"
    [ -n "${frpc_loop_pid}" ] && kill "${frpc_loop_pid}" 2>/dev/null || true
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_http() {
    local url="$1" timeout_s="$2" name="$3" elapsed=0
    until curl -sf "${url}" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "${elapsed}" -ge "${timeout_s}" ]; then
            echo "[tp2-mw-start] timeout waiting for ${name}: ${url}" >&2
            return 1
        fi
    done
    echo "[tp2-mw-start] ${name} ready after ${elapsed}s"
}

start_frpc_retry_loop() {
    if [ "${ENABLE_FRP}" != "1" ]; then
        echo "[tp2-mw-start] frp disabled"
        return 0
    fi
    if [ -z "${FRPC_CONFIG}" ] || [ ! -f "${FRPC_CONFIG}" ]; then echo "[tp2-mw-start] missing FRPC_CONFIG=${FRPC_CONFIG}" >&2; exit 1; fi
    if [ ! -x "${FRPC_BIN}" ]; then echo "[tp2-mw-start] missing frpc: ${FRPC_BIN}" >&2; exit 1; fi
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
    echo "[tp2-mw-start] frpc retry loop pid=${frpc_loop_pid}"
}

visible_devices_for_worker() {
    local worker_index="$1"
    local start_gpu=$((worker_index * GPUS_PER_WORKER))
    local end_gpu=$((start_gpu + GPUS_PER_WORKER - 1))
    local out="" gpu
    for ((gpu=start_gpu; gpu<=end_gpu; gpu++)); do
        if [ -n "${out}" ]; then out="${out},"; fi
        out="${out}${gpu}"
    done
    echo "${out}"
}

echo "[tp2-mw-start] project=${PROJECT_DIR}"
echo "[tp2-mw-start] model=${MODEL_PATH}"
echo "[tp2-mw-start] pt=${PT_PATH}"
echo "[tp2-mw-start] token_trace_dir=${O5_TOKEN_TRACE_DIR}"
echo "[tp2-mw-start] backbone=${BACKBONE_DIR}"
echo "[tp2-mw-start] workers=${NUM_WORKERS} gpus_per_worker=${GPUS_PER_WORKER} total_gpus=$((NUM_WORKERS * GPUS_PER_WORKER))"
echo "[tp2-mw-start] llm_cache=${O5_LLM_CACHE} spmd_heartbeat=${O5_SPMD_HEARTBEAT_INTERVAL}"
echo "[tp2-mw-start] deployment_mode=${O5_DEPLOY_MODE:-tp2} llm_graph=${O5_LLM_GRAPH:-1}"
echo "[tp2-mw-start] gateway=https://${GATEWAY_HOST}:${GATEWAY_PORT} internal=:${GATEWAY_INTERNAL_PORT}"

"${PYTHON}" gateway.py \
    --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}" --internal-port "${GATEWAY_INTERNAL_PORT}" \
    --https --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem \
    > "${LOG_DIR}/gateway.log" 2>&1 &
pids+=("$!")
wait_http "http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/health" 120 "gateway-internal"

backend_urls=()
worker_endpoints=()

for ((i=0; i<NUM_WORKERS; i++)); do
    backend_port=$((BACKEND_BASE_PORT + i))
    worker_port=$((WORKER_BASE_PORT + i))
    backend_url="http://${BACKEND_HOST}:${backend_port}"
    worker_endpoint="${WORKER_HOST}:${worker_port}"
    visible_devices="$(visible_devices_for_worker "${i}")"
    master_port=$((TORCHRUN_MASTER_BASE_PORT + i))

    backend_urls+=("${backend_url}")
    worker_endpoints+=("${worker_endpoint}")

    echo "[tp2-mw-start] backend ${i}: cuda_visible=${visible_devices} master_port=${master_port} url=${backend_url}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
    O5_TORCHRUN_MASTER_PORT="${master_port}" \
    "${PROJECT_DIR}/core/deploy/launch_tp2.sh" \
        --host "${BACKEND_HOST}" --port "${backend_port}" \
        --model-path "${MODEL_PATH}" --pt-path "${PT_PATH}" \
        > "${LOG_DIR}/backend_tp2_${i}.log" 2>&1 &
    pids+=("$!")

    if [ "${BACKEND_START_STAGGER_SECONDS}" -gt 0 ] && [ "$((i + 1))" -lt "${NUM_WORKERS}" ]; then
        echo "[tp2-mw-start] waiting ${BACKEND_START_STAGGER_SECONDS}s before launching next tp2 backend"
        sleep "${BACKEND_START_STAGGER_SECONDS}"
    fi
done

for ((i=0; i<NUM_WORKERS; i++)); do
    wait_http "${backend_urls[$i]}/health" 1200 "backend-tp2-${i}"
done

for ((i=0; i<NUM_WORKERS; i++)); do
    worker_port=$((WORKER_BASE_PORT + i))
    backend_url="${backend_urls[$i]}"
    worker_endpoint="${worker_endpoints[$i]}"
    visible_devices="$(visible_devices_for_worker "${i}")"
    start_gpu=$((i * GPUS_PER_WORKER))

    echo "[tp2-mw-start] worker ${i}: endpoint=${worker_endpoint} backend=${backend_url} gpu_group=${visible_devices}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" "${PYTHON}" worker.py \
        --host "${WORKER_HOST}" --port "${worker_port}" \
        --gpu-id "${start_gpu}" --backend-server-url "${backend_url}" \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &
    pids+=("$!")
done

for ((i=0; i<NUM_WORKERS; i++)); do
    wait_http "http://${worker_endpoints[$i]}/health" 120 "worker-${i}"
done

for ((i=0; i<NUM_WORKERS; i++)); do
    worker_endpoint="${worker_endpoints[$i]}"
    worker_id="${WORKER_ID_PREFIX}-${i}"
    visible_devices="$(visible_devices_for_worker "${i}")"
    gpu_group="${WORKER_GPU_GROUP_PREFIX}-${visible_devices//,/-}"
    payload="{\"endpoint\":\"${worker_endpoint}\",\"gpu_group\":\"${gpu_group}\",\"labels\":{\"model\":\"o5\",\"runtime\":\"tp2-cctl\",\"worker_index\":\"${i}\",\"cuda_visible_devices\":\"${visible_devices}\"}}"
    curl -sf -X PUT -H "content-type: application/json" --data "${payload}" \
        "http://127.0.0.1:${GATEWAY_INTERNAL_PORT}/internal/workers/${worker_id}" >/dev/null
    echo "[tp2-mw-start] registered ${worker_id} endpoint=${worker_endpoint} gpu_group=${gpu_group}"
done

curl -sk "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null
echo "[tp2-mw-start] service ready with ${NUM_WORKERS} TP2 workers"

start_frpc_retry_loop

while true; do
    for pid in "${pids[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[tp2-mw-start] child exited: ${pid}" >&2
            exit 1
        fi
    done
    if [ -n "${frpc_loop_pid}" ] && ! kill -0 "${frpc_loop_pid}" 2>/dev/null; then
        echo "[tp2-mw-start] frpc loop exited" >&2
        exit 1
    fi
    sleep 5
done
