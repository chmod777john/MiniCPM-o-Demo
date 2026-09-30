#!/usr/bin/env bash
# 在一个 Cybertron Job 内启动 O5 API v3 副本池并运行固定样例吞吐测试。

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PT_PATH="${PT_PATH:?set PT_PATH to the full O5 .pt}"
BACKBONE_DIR="${BACKBONE_DIR:-}"
TRAINING_DATA="${TRAINING_DATA:?set TRAINING_DATA to JSON or JSONL}"
DATA_ROOT="${DATA_ROOT:-$(dirname "${TRAINING_DATA}")}"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:?set single_eager, single_opt, or tp2}"
GPU_GROUPS="${GPU_GROUPS:?set GPU groups such as 0 or 0,1 or 0;1;...}"
REPLICA_COUNT="${REPLICA_COUNT:?set expected replica count}"
REPETITIONS="${REPETITIONS:-10}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${DEPLOYMENT_MODE}-${REPLICA_COUNT}rep}"
RUNTIME_DIR="/tmp/o5-throughput/${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/o5-throughput-result/${RUN_ID}}"
INTERNAL_SETTINGS="${O5_API_INTERNAL_SETTINGS_PATH:-${PROJECT_DIR}/configs/fc_deployment/o5_api_internal_settings.json}"
GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"

RUNTIME_PYTHON="$(
    python3 - "${INTERNAL_SETTINGS}" <<'PY'
import json
import sys
from pathlib import Path

settings = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(settings["runtime_python"])
PY
)"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

service_pid=""
monitor_pid=""

cleanup() {
    local exit_code=$?
    if [ -n "${monitor_pid}" ] && kill -0 "${monitor_pid}" 2>/dev/null; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [ -n "${service_pid}" ] && kill -0 "${service_pid}" 2>/dev/null; then
        kill -TERM "${service_pid}" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "${service_pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${service_pid}" 2>/dev/null; then
            kill -KILL "${service_pid}" 2>/dev/null || true
        fi
        wait "${service_pid}" 2>/dev/null || true
    fi
    if [ -d "${RUNTIME_DIR}/logs" ]; then
        mkdir -p "${OUTPUT_DIR}/runtime_logs"
        cp -a "${RUNTIME_DIR}/logs/." "${OUTPUT_DIR}/runtime_logs/" \
            2>/dev/null || true
    fi
    echo "[O5_THROUGHPUT_CLEANUP] run_id=${RUN_ID} exit_code=${exit_code}"
    return "${exit_code}"
}
trap cleanup EXIT INT TERM

{
    echo "run_id=${RUN_ID}"
    echo "deployment_mode=${DEPLOYMENT_MODE}"
    echo "gpu_groups=${GPU_GROUPS}"
    echo "replica_count=${REPLICA_COUNT}"
    echo "repetitions=${REPETITIONS}"
    echo "pt_path=${PT_PATH}"
    echo "backbone_dir=${BACKBONE_DIR}"
    echo "training_data=${TRAINING_DATA}"
    echo "data_root=${DATA_ROOT}"
    echo "git_head=$(git rev-parse HEAD)"
    echo "git_diff_sha256=$(git diff --no-ext-diff | sha256sum | awk '{print $1}')"
    echo "pt_size_bytes=$(stat -c %s "${PT_PATH}")"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${OUTPUT_DIR}/run_manifest.env"

"${PROJECT_DIR}/run_o5_replica_pool.sh" \
    --pt-path "${PT_PATH}" \
    ${BACKBONE_DIR:+--backbone-dir "${BACKBONE_DIR}"} \
    --deployment-mode "${DEPLOYMENT_MODE}" \
    --gpu-groups "${GPU_GROUPS}" \
    --runtime-dir "${RUNTIME_DIR}" \
    --gateway-port "${GATEWAY_PORT}" \
    --gateway-internal-port "${GATEWAY_INTERNAL_PORT}" \
    > "${OUTPUT_DIR}/supervisor.log" 2>&1 &
service_pid=$!

ready_path="${RUNTIME_DIR}/READY.json"
for _ in $(seq 1 900); do
    if [ -f "${ready_path}" ]; then
        break
    fi
    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "[benchmark-job] service exited before ready" >&2
        tail -n 200 "${OUTPUT_DIR}/supervisor.log" >&2 || true
        for log_path in "${RUNTIME_DIR}"/logs/*.log; do
            [ -f "${log_path}" ] || continue
            echo "[benchmark-job] tail ${log_path}" >&2
            tail -n 200 "${log_path}" >&2 || true
        done
        exit 1
    fi
    sleep 2
done
if [ ! -f "${ready_path}" ]; then
    echo "[benchmark-job] timeout waiting for ${ready_path}" >&2
    tail -n 200 "${OUTPUT_DIR}/supervisor.log" >&2 || true
    exit 1
fi

(
    echo "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_percent,host_memory_used_bytes"
    while kill -0 "${service_pid}" 2>/dev/null; do
        timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        host_used="$(
            free -b | awk '/^Mem:/ {print $3}'
        )"
        nvidia-smi \
            --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits \
            | while IFS= read -r row; do
                echo "${timestamp},${row},${host_used}"
            done
        sleep 1
    done
) > "${OUTPUT_DIR}/resource_metrics.csv" &
monitor_pid=$!

"${RUNTIME_PYTHON}" \
    "${PROJECT_DIR}/scripts/benchmark_o5_api_throughput.py" \
    "${TRAINING_DATA}" \
    --data-root "${DATA_ROOT}" \
    --base-url "http://127.0.0.1:${GATEWAY_PORT}" \
    --replica-count "${REPLICA_COUNT}" \
    --repetitions "${REPETITIONS}" \
    --output-dir "${OUTPUT_DIR}/benchmark"

python3 - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = json.loads((output_dir / "benchmark" / "summary.json").read_text(encoding="utf-8"))
gpu_peak = {}
host_peak = 0
with (output_dir / "resource_metrics.csv").open(newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        gpu = int(row["gpu_index"].strip())
        used = float(row["memory_used_mib"].strip())
        gpu_peak[gpu] = max(gpu_peak.get(gpu, 0.0), used)
        host_peak = max(host_peak, int(row["host_memory_used_bytes"].strip()))
resource_summary = {
    "gpu_peak_memory_mib": gpu_peak,
    "host_peak_memory_bytes": host_peak,
}
(output_dir / "resource_summary.json").write_text(
    json.dumps(resource_summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    "[O5_THROUGHPUT_JOB_RESULT] "
    f"success={summary['success_count']}/{summary['total_session_count']} "
    f"sessions_per_sec={summary['sessions_per_sec']:.6f} "
    f"host_peak_bytes={host_peak} gpu_peak_mib={gpu_peak}"
)
PY

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${OUTPUT_DIR}/run_manifest.env"
