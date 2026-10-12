#!/usr/bin/env bash
# 对外 API 部署：开发机上的 API Gateway + 控制面板（Worker 是 deploy 池里各自的 cctl job，由面板起停）。
#
# 用法：scripts/start_api_gateway.sh <env 文件>
# env 文件放仓库外（600 权限），至少包含：
#   CONTROL_PANEL_PASSWORD   控制面板密码（不设则面板一律 503）
#   CONTROL_PLANE_ROOT       控制面状态与 Worker 运行目录的根（必须在共享存储上：Worker job 要读写它）
#   API_GATEWAY_PORT         对外 HTTPS 端口（本机；再由 frpc 暴露到公网），默认 10029
#   API_GATEWAY_INTERNAL_PORT  内部注册端口（只绑 127.0.0.1），默认 10050
#   API_PUBLIC_BASE_URL      面板上展示的公网地址，例如 https://47.95.219.248:7029
#   CCTL_BIN / CCTL_PROFILE / HOME  控制面调 cctl 起停 Worker 用（cctl 读 $HOME 下的登录态）
set -euo pipefail

ENV_FILE="${1:?usage: $0 <env file>}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_python"])' \
      "${REPO}/configs/fc_deployment/o5_api_internal_settings.json")"
: "${CONTROL_PLANE_ROOT:?CONTROL_PLANE_ROOT is required}"
API_GATEWAY_PORT="${API_GATEWAY_PORT:-10029}"
API_GATEWAY_INTERNAL_PORT="${API_GATEWAY_INTERNAL_PORT:-10050}"

mkdir -p "${CONTROL_PLANE_ROOT}/gateway"
# 会话录音落在控制面目录下，不写进仓库。
cat > "${CONTROL_PLANE_ROOT}/gateway/service_config.json" <<EOF
{"service": {"data_dir": "${CONTROL_PLANE_ROOT}/gateway/data"}}
EOF
export O5_DEMO_CONFIG_PATH="${CONTROL_PLANE_ROOT}/gateway/service_config.json"
# 只放行 /e/、/api-docs、/control、/health；O45 老页面与会改服务状态的旧接口不对外。
export GATEWAY_API_ONLY=1
# 连 Pod（10.x）的健康检查与 WebSocket 不能走本机代理。
export NO_PROXY="10.0.0.0/8,127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

cd "${REPO}"
echo "[api-gateway] repo=${REPO} commit=$(git -C "${REPO}" rev-parse --short HEAD) port=${API_GATEWAY_PORT} root=${CONTROL_PLANE_ROOT}"
exec "${PY}" gateway.py \
    --host 127.0.0.1 --port "${API_GATEWAY_PORT}" \
    --internal-host 127.0.0.1 --internal-port "${API_GATEWAY_INTERNAL_PORT}" \
    --https --ssl-certfile "${REPO}/certs/cert.pem" --ssl-keyfile "${REPO}/certs/key.pem" \
    --control-plane-root "${CONTROL_PLANE_ROOT}"
