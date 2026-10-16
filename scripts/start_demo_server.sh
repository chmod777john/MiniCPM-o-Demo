#!/usr/bin/env bash
# Demo 服务（独立于 API）：页面 + 各 Demo 自己的后端；页面上选 API 端点，经协议 v4 访问。
#
# 用法：scripts/start_demo_server.sh <env 文件>
# env 文件放仓库外（600 权限），按需包含：
#   DEMO_PORT                本机 HTTPS 端口（再由 frpc 暴露到公网），默认 10032
#   DEMO_API_ENDPOINTS       页面可选的 API 端点 JSON：[{"name":"f9t4","url":"https://<host>:<port>/e/f9t4"}]
#   DEMO_DEFAULT_API         首次打开时选中的端点（缺省取列表第一个）
#   DEMO_API_SERVICE         首次打开时的 API 服务地址（https://<host>:<port>，页面上可改）；页面读其 /overview/api/state
#                            把全部端点（一个端点 = 一个模型）列进下拉框。缺省取 DEMO_DEFAULT_API 所在的服务
#   DEMO_DEFAULT_PRESET      首次打开时的 preset id
#   DEMO_AUDIO_ROOTS         允许读取参考音频的目录（: 分隔）；不设则不读任何音频
#   DEMO_BOARD_CASE_FOLDER   board 默认 system 取自的训练数据目录
#   SUBAGENT_CLAUDE_BIN / SUBAGENT_CLAUDE_SETTINGS  subagent 后台 Claude Code（不设则 subagent 503）
set -euo pipefail

ENV_FILE="${1:?usage: $0 <env file>}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_python"])' \
      "${REPO}/configs/fc_deployment/o5_api_internal_settings.json")"
DEMO_PORT="${DEMO_PORT:-10032}"

cd "${REPO}"
echo "[demo-server] repo=${REPO} commit=$(git -C "${REPO}" rev-parse --short HEAD) port=${DEMO_PORT}"
exec "${PY}" demo_server.py --host 127.0.0.1 --port "${DEMO_PORT}" --https
