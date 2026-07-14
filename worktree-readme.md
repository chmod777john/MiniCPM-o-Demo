# Worktree: o5-tp2-encapsulation-2026-07-14

- Created at: 2026-07-14
- Source worktree: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-multimode-deploy-2026-07-13`
- Source branch: `wt/o5-multimode-deploy-wkq-deploy-2026-07-13`
- Source commit: `98b8d66` (`Mirror spmd generator calls`)
- New branch: `wt/o5-tp2-encapsulation-2026-07-14`

## Purpose

Explore how to encapsulate the O5 TP2/distributed execution path so that, relative to the existing optimized single-card path, enabling two-card inference does not require changing unrelated modeling or business logic.

The intended discussion target is a cleaner boundary where TP2-specific process/rank coordination is isolated behind model/runtime construction or LLM-forward-level wrappers, while the rest of the demo continues to call the same public model APIs.
