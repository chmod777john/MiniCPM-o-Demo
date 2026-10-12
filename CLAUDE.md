> 📍 **本 CLAUDE.md** — 所属目录 `/user/sunweiyue/lib/MiniCPM-o-Demo-modelbest/` · 项目级（入口索引）

# MiniCPM-o Demo 仓库

## 最高优先级：推理必须忠实反映 checkpoint 能力

Demo 是模型能力的测量仪器，不是补偿器。**禁止**为了展示效果改推理语义：
加 logit bias、强制 token、改 prompt、补写或吞掉模型生成的协议 token 等。
模型效果不好时，结论回到训练 / 数据 / checkpoint。
全文：`.cursor/rules/model-faithful-inference.mdc`（Claude 不会自动读 `.cursor/`，改推理代码前必读）。

## API 与 Demo 是两件事（2026-10-12 起）

放在一个仓库只是图方便：**两侧只通过协议耦合，Python 代码互不 import**（守卫测试 `tests/test_api_demo_boundary.py`）。

| | API | Demo |
|---|---|---|
| 代码 | `gateway.py`、`gateway_modules/`、`control_plane/`、`worker.py`、`py_backend/`、`core/`、`modeling/`、`clients/` | `demo_server.py`、`demo_gallery/`、`demos/`、`static/demo-gallery/` |
| 协议 | 对外只认 **v4**（`docs/fc-duplex/semantic-realtime-api-v4.md`）：音频只收 base64 WAV，不收服务端路径；模型口径（预算、调度、tts_pad、profile）由部署绑定 | 页面上选 API 端点，按 v4 直连；会话前查 `<端点>/v1/info`；音频路径由 Demo 服务读成 base64 |
| v3 | 只在评测副本池里保留（`scripts/run_o5_replica_pool.py` 设 `FC_API_ACCEPT_V3=1`，评测锁步客户端不动） | — |

## 部署

```text
开发机（廊坊 devbox）
  公网 7029 → frpc → 127.0.0.1:10029  API Gateway + 控制面板   scripts/start_api_gateway.sh <api.env>
       /e/<端点>/v1/realtime（WS）、/e/<端点>/v1/info、/api-docs、/overview（公开只读状态页）、/control（密码）
  公网 7032 → frpc → 127.0.0.1:10032  Demo 服务                 scripts/start_demo_server.sh <demo.env>
deploy 资源池：每个 Worker 一个 cctl job（2×A100 TP2，scripts/o5_worker_job.py），由控制面板起停
```

- **起停 Worker、扩容、驱逐、强制关闭、换模型：在控制面板 `https://47.95.219.248:7029/control` 上操作**，
  不要手工 `cctl job create`。面板状态与 Worker 运行目录在 `/user/sunweiyue/runtime/o5_api_control_plane/`。
- 两个 env 文件（含面板密码、subagent 的 Claude Code 配置）在 `/user/sunweiyue/runtime/o5_api_deploy/`，600 权限，不进仓库。
- 重启 API Gateway 不影响在跑的 Worker job：重启后从面板状态与就绪文件把它们重新登记回端点（进行中的会话会断）。
- 两个进程都用 `setsid nohup … &` 脱离会话启动，日志分别在 `o5_api_control_plane/gateway.log`、`o5_api_deploy/demo_server.log`。

## 要新加或修改一个 Demo？

**先读 `demos/CLAUDE.md`**（硬规则与流程），API 细节看 `demos/README.md`。

一句话：一个 Demo = `demos/<id>/` 一个文件夹，从 `demos/_template/` 复制起步。
**不许为某个 Demo 修改 `demo_server.py`、`demo_gallery/`、`static/demo-gallery/`**，
有测试强制。

## O45 已停止维护（2026-10-10 起）

本仓库只维护 **O5**。O45（`modeling/o45/`、`O45FcDuplexModelAdapter`、`o45_fc` tokenizer target）
的代码保留但冻结：**不再修 bug、不再加新功能、不再为它补测试**。

- 新协议能力只做 O5。O45 收到不支持的输入时直接报明确错误，不要为 O45 补实现。
- **Stateless Resume（断点续传）只有 O45 实现过**，O5 没有（`supports_stateless_resume=False`），
  因此续传整体随 O45 冻结。`docs/fc-duplex/resumable-generation-api.md` 只作历史参考，不再演进。
- 改共享代码时不必再顾及 O45 路径的行为；但也不要顺手删它，删除另行决定。

## 仓库事实

- **线上 API Gateway、Demo 服务、Worker job 都直接跑本目录**：前端文件保存即上线；Worker job 启动时读的也是这里的代码。
  半成品用 `git worktree` 开发，详见 `demos/CLAUDE.md`。
- **唯一远端是 `codehub`**（华为 codehub），分支 upstream 都指 `codehub/*`，裸 `git push` 即可。
  原 codeup 远端（`git@codeup-modelbest:modelbest/multi-modal/MiniCPM-o-Demo.git`）
  已禁止推送，2026-10-10 已从本仓库移除，不要再加回来。
- 测试 Python 环境：`/user/sunweiyue/runtime/o5_eval/venv-o5-accel/bin/python`。
