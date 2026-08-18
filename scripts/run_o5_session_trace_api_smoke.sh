#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
MODEL_PATH="${MODEL_PATH:-}"
PT_PATH="${PT_PATH:-}"
VIDEO_PATH="${VIDEO_PATH:-}"
PROMPT_WAV="${PROMPT_WAV:-}"
MAX_UNITS="${MAX_UNITS:-8}"
RUN_DIR="${RUN_DIR:-/user/weihongliang/o5_session_trace_replay_runs/api-smoke-${MAX_UNITS}u-20260818}"

GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"
BACKEND_PORT="${BACKEND_PORT:-22510}"
WORKER_PORT="${WORKER_PORT:-22410}"
START_TIMEOUT_S="${START_TIMEOUT_S:-1200}"
EVENT_TIMEOUT_S="${EVENT_TIMEOUT_S:-600}"
PYTHON="${VENV_DIR}/bin/python"

for path_name in MODEL_PATH PT_PATH VIDEO_PATH PROMPT_WAV; do
    path_value="${!path_name}"
    if [ -z "${path_value}" ] || [ ! -e "${path_value}" ]; then
        echo "[api-trace-smoke] missing ${path_name}: ${path_value}" >&2
        exit 2
    fi
done
if [ ! -x "${PYTHON}" ]; then
    echo "[api-trace-smoke] missing python: ${PYTHON}" >&2
    exit 2
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}/service_logs"
cd "${PROJECT_DIR}"

export PROJECT_DIR VENV_DIR MODEL_PATH PT_PATH
export GATEWAY_PORT GATEWAY_INTERNAL_PORT BACKEND_PORT WORKER_PORT
export ENABLE_FRP=0
export LOG_DIR="${RUN_DIR}/service_logs"
export O5_TOKEN_TRACE_DIR="${PROJECT_DIR}/data/sessions"
export O5_SESSION_TRACE_MODE=replay

echo "[api-trace-smoke] run_dir=${RUN_DIR}"
echo "[api-trace-smoke] max_units=${MAX_UNITS}"
echo "[api-trace-smoke] trace_dir=${O5_TOKEN_TRACE_DIR}"

bash scripts/start_o5_cctl_service.sh >"${RUN_DIR}/service.log" 2>&1 &
service_pid=$!

cleanup() {
    kill "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + START_TIMEOUT_S))
until curl -skf "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "[api-trace-smoke] service exited during startup" >&2
        tail -n 240 "${RUN_DIR}/service.log" >&2 || true
        exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "[api-trace-smoke] service startup timed out after ${START_TIMEOUT_S}s" >&2
        tail -n 240 "${RUN_DIR}/service.log" >&2 || true
        exit 1
    fi
    sleep 2
done
echo "[api-trace-smoke] service ready"

"${PYTHON}" -B tools/o5trace/api_canonical_video_probe.py \
    --url "https://127.0.0.1:${GATEWAY_PORT}" \
    --insecure \
    --video "${VIDEO_PATH}" \
    --prompt-wav "${PROMPT_WAV}" \
    --out-dir "${RUN_DIR}/api" \
    --max-units "${MAX_UNITS}" \
    --event-timeout-s "${EVENT_TIMEOUT_S}" \
    --include-events \
    | tee "${RUN_DIR}/probe.log"

session_id="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["session_created"]["session_id"])' "${RUN_DIR}/api/summary.json")"
session_dir="${O5_TOKEN_TRACE_DIR}/${session_id}"

deadline=$((SECONDS + 60))
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if [ -s "${session_dir}/stream.jsonl" ] \
        && [ -s "${session_dir}/model_trace.jsonl" ] \
        && [ -s "${session_dir}/trace_manifest.json" ]; then
        completed="$("${PYTHON}" -c 'import json, sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8")).get("completed"))).lower())' "${session_dir}/trace_manifest.json")"
        ended="$("${PYTHON}" -c 'import json, sys; print(str(json.load(open(sys.argv[1], encoding="utf-8")).get("ended_at") is not None).lower())' "${session_dir}/meta.json" 2>/dev/null || true)"
        if [ "${completed}" = "true" ] && [ "${ended}" = "true" ]; then
            break
        fi
    fi
    sleep 1
done

"${PYTHON}" - "${session_dir}" "${RUN_DIR}/api/summary.json" "${MAX_UNITS}" <<'PY'
import json
import sys
from pathlib import Path

session_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected_units = int(sys.argv[3])
required = ["meta.json", "stream.jsonl", "model_trace.jsonl", "trace_manifest.json"]
missing = [name for name in required if not (session_dir / name).is_file() or not (session_dir / name).stat().st_size]
if missing:
    raise SystemExit(f"missing or empty session artifacts: {missing}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads((session_dir / "trace_manifest.json").read_text(encoding="utf-8"))
meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
tensors = list((session_dir / "trace_tensors").glob("*.pt"))
units = len(summary.get("units") or [])
if units != expected_units:
    raise SystemExit(f"unit count mismatch: expected={expected_units} actual={units}")
if manifest.get("completed") is not True:
    raise SystemExit("trace manifest is not completed")
if meta.get("ended_at") is None:
    raise SystemExit("gateway session recording is not closed")
if not tensors:
    raise SystemExit("trace_tensors contains no tensor payloads")

events = sum(1 for line in (session_dir / "model_trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
stream_frames = sum(1 for line in (session_dir / "stream.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
print(json.dumps({
    "session_id": meta.get("session_id"),
    "session_dir": str(session_dir),
    "units": units,
    "model_trace_events": events,
    "stream_frames": stream_frames,
    "trace_tensors": len(tensors),
    "completed": manifest.get("completed"),
}, ensure_ascii=False, indent=2))
PY

echo "[api-trace-smoke] passed session_dir=${session_dir}"
