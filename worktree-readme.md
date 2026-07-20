# Worktree: o5 TP2 CUDA Graph profile

- Created at: 2026-07-20
- Created from branch: `wt/o5-pure-moe-profile-2026-07-14`
- Created from commit: `9f0c772e4176c3738b97314c58e6c9f7c265ba10`
- New branch: `wt/o5-tp2-graph-profile-2026-07-20`
- Purpose: isolate experiments for Qwen3.5-MoE tensor-parallel decode with CUDA Graph. The immediate question is whether `tp_plan="auto"` + one-token decode can be graph-captured on two A100 GPUs, and whether it improves over TP2 eager decode.
- Scope: profiling scripts and report notes only. Do not change demo serving code.
