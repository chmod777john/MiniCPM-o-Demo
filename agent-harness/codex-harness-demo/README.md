# Codex Harness Demo

一个最小 harness demo：左侧模拟 `o45 -> Codex` 的对话和工具调用过程，右侧实时预览 Codex 修改的 session-local `index.html`。

## Codex 来源

这个 demo **不会使用 PATH 里的全局 `codex`**。启动时只接受：

1. `CODEX_BIN=/abs/path/to/codex`
2. 或自动探测本地 Codex worktree / 源码构建产物：
   - `/user/weihongliang/codex-wt-harness-local-2026-06-10/codex-rs/target/debug/codex`
   - `/user/weihongliang/codex-wt-harness-local-2026-06-10/codex-rs/target/release/codex`
   - `/user/weihongliang/codex/codex-rs/target/debug/codex`
   - `/user/weihongliang/codex/codex-rs/target/release/codex`

默认 Python SDK 也从 worktree 读取：

```text
/user/weihongliang/codex-wt-harness-local-2026-06-10/sdk/python/src
```

如果还没构建，先确认 Rust 可用，然后运行：

```bash
/user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/agent-harness/codex-harness-demo/scripts/build-local-codex.sh
```

然后启动：

```bash
cd /user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/agent-harness/codex-harness-demo
python3 server.py
```

打开：

```text
http://127.0.0.1:8765
```

## MVP 行为

- 每个浏览器 session 创建独立 workspace：
  - `.data/sessions/{session_id}/workspace/index.html`
- Codex 的 `cwd` 指向该 workspace。
- prompt 固定要求 Codex 只编辑 `index.html`。
- 后端 watch `index.html`，变更后发送 `preview.updated`。
- 前端右侧 iframe 使用 `/preview/{session_id}/index.html?v=N` 实时刷新。

## 安全默认值

- `HARNESS_CODEX_SANDBOX=auto` 是默认值：
  - 如果本机 `bubblewrap` 可用，Codex thread/turn 使用 `workspace-write` sandbox。
  - 如果 `bubblewrap` 在当前容器里不可用，demo 会降级为 `Sandbox.full_access`，并在 `/api/status`、SSE 和 UI 中显示警告。
- `web_search` 固定为 `disabled`。
- `sandbox_workspace_write.network_access=false`。
- approval mode 使用 `deny_all`，避免自动批准 sandbox 升权。

也可以显式指定：

```bash
HARNESS_CODEX_SANDBOX=workspace-write python3 server.py
HARNESS_CODEX_SANDBOX=full-access python3 server.py
```

这是 demo 级别的约束，不是生产隔离边界。生产版还需要更严格的 workspace 清理、静态文件白名单、进程生命周期管理和资源限制。
