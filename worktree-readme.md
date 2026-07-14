# Worktree 说明

- 创建时间：2026-07-14
- 来源仓库：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30`
- 来源分支：`wt/o5-inference-refactor-2026-06-30`
- 来源 commit：`07fcb1a421f5525689eb600655192eec8adcb6fb`
- 当前分支：`wt/o5-pure-moe-profile-2026-07-14`
- 当前路径：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-pure-moe-profile-2026-07-14`

## 目的

这个 worktree 用于对 o5 backbone 里的纯 Qwen3.5 MoE 推理过程做性能 profiling。
目标是尽量排除 MiniCPM-o 外层、音频、TTS、duplex、服务框架等因素，只观察 LLM/MoE 本体在 prefill 和 decode 阶段的耗时。

默认只读使用以下模型目录，不修改该目录内容：

`/user/weihongliang/wangkaiqi/o5_backbone_hf`

