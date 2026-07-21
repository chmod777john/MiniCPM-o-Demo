# Worktree: o5-no-fc-speedup-tp2

- Created at: 2026-07-19
- Source worktree: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-tp2-llm-wrapper-2026-07-15`
- Source branch: `wt/o5-tp2-llm-wrapper-2026-07-15`
- Source commit: `84e224b` (`Restore direct nonstream chat path`)
- New worktree: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2`
- New branch: `o5-no-fc-speedup-tp2`

## Purpose

Carry the TP2 LLM-wrapper serving path forward as the two-card counterpart of `o5-no-fc-speedup`.

This worktree starts from the TP2 wrapper branch, then brings in the service/frontend updates already used by the single-card speedup line: multi-worker launcher support, worker GPU isolation, staged backend startup, omni sharing safeguards, 32k default MaxKV, larger recording uploads, and longer video-file share handling.

The intent is to preserve the TP2 model/runtime boundary while keeping the surrounding demo behavior aligned with the current speedup branch.
