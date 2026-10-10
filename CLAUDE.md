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

## 仓库事实

- **线上 Demo 服务直接跑本目录**：前端文件保存即上线。半成品用 `git worktree` 开发，
  详见 `demos/CLAUDE.md` 末节。
- **推送远端是 `codehub`**：`git push codehub <分支>`。`origin`（codeup）已禁止推送；
  分支的 upstream 仍指 `origin/*`，所以 `git status` 里的「ahead N」是假数字，
  查真实位置用 `git ls-remote codehub <分支>`。
- 测试 Python 环境：`/user/sunweiyue/runtime/o5_eval/venv-o5-accel/bin/python`。
