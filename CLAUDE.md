> 📍 **本 CLAUDE.md** — 所属目录 `/user/sunweiyue/lib/MiniCPM-o-Demo-modelbest/` · 项目级（入口索引）

# MiniCPM-o Demo 仓库

## 最高优先级：推理必须忠实反映 checkpoint 能力

Demo 是模型能力的测量仪器，不是补偿器。**禁止**为了展示效果改推理语义：
加 logit bias、强制 token、改 prompt、补写或吞掉模型生成的协议 token 等。
模型效果不好时，结论回到训练 / 数据 / checkpoint。
全文：`.cursor/rules/model-faithful-inference.mdc`（Claude 不会自动读 `.cursor/`，改推理代码前必读）。

## 要新加或修改一个 Demo？

**先读 `demos/CLAUDE.md`**（硬规则与流程），API 细节看 `demos/README.md`。

一句话：一个 Demo = `demos/<id>/` 一个文件夹，从 `demos/_template/` 复制起步。
**不许为某个 Demo 修改 `gateway.py`、`demo_gallery/`、`static/demo-gallery/`**，
有测试强制。

## O45 已停止维护（2026-10-10 起）

本仓库只维护 **O5**。O45（`modeling/o45/`、`O45FcDuplexModelAdapter`、`o45_fc` tokenizer target）
的代码保留但冻结：**不再修 bug、不再加新功能、不再为它补测试**。

- 新协议能力只做 O5。O45 收到不支持的输入时直接报明确错误，不要为 O45 补实现。
- **Stateless Resume（断点续传）只有 O45 实现过**，O5 没有（`supports_stateless_resume=False`），
  因此续传整体随 O45 冻结。`docs/fc-duplex/resumable-generation-api.md` 只作历史参考，不再演进。
- 改共享代码时不必再顾及 O45 路径的行为；但也不要顺手删它，删除另行决定。

## 仓库事实

- **线上 Demo 服务直接跑本目录**：前端文件保存即上线。半成品用 `git worktree` 开发，
  详见 `demos/CLAUDE.md` 末节。
- **唯一远端是 `codehub`**（华为 codehub），分支 upstream 都指 `codehub/*`，裸 `git push` 即可。
  原 codeup 远端（`git@codeup-modelbest:modelbest/multi-modal/MiniCPM-o-Demo.git`）
  已禁止推送，2026-10-10 已从本仓库移除，不要再加回来。
- 测试 Python 环境：`/user/sunweiyue/runtime/o5_eval/venv-o5-accel/bin/python`。
