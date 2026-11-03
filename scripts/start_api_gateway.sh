#!/usr/bin/env bash
# 对外 API 部署：开发机上的 API Gateway（Worker 由部署平台或内部注册接口管理）。
#
# 用法：scripts/start_api_gateway.sh <env 文件>
# env 文件放仓库外（600 权限），可包含：
#   API_GATEWAY_PORT         对外 HTTPS 端口（本机；再由 frpc 暴露到公网），默认 10029
#   API_GATEWAY_INTERNAL_PORT  内部注册端口（只绑 127.0.0.1），默认 10050
#   O5_SESSION_DATA_DIR        可选，Session 原始收发帧录制根目录；
#                              例如 /user/weihongliang/o-sessions
set -euo pipefail

ENV_FILE="${1:?usage: $0 <env file>}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_python"])' \
      "${REPO}/configs/fc_deployment/o5_api_internal_settings.json")"
API_GATEWAY_PORT="${API_GATEWAY_PORT:-10029}"
API_GATEWAY_INTERNAL_PORT="${API_GATEWAY_INTERNAL_PORT:-10050}"

# 只放行模型目录、命名实时端点与健康检查；应用与管理 API 不对外。
export GATEWAY_API_ONLY=1
# 连 Pod（10.x）的健康检查与 WebSocket 不能走本机代理。
export NO_PROXY="10.0.0.0/8,127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

cd "${REPO}"
echo "[api-gateway] repo=${REPO} commit=$(git -C "${REPO}" rev-parse --short HEAD) port=${API_GATEWAY_PORT}"
exec "${PY}" gateway.py \
    --host 127.0.0.1 --port "${API_GATEWAY_PORT}" \
    --internal-host 127.0.0.1 --internal-port "${API_GATEWAY_INTERNAL_PORT}" \
    --https --ssl-certfile "${REPO}/certs/cert.pem" --ssl-keyfile "${REPO}/certs/key.pem"
