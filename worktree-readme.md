# Worktree: o5-no-fc-speedup-tp2-session-trace-replay-2026-08-18

- Created: 2026-08-18
- Base worktree: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2`
- Base branch: `o5-no-fc-speedup-tp2`
- Base commit: `7224ea08f37bbfbb1a464cbd15849152be4f5594`
- New branch: `wt/o5-no-fc-speedup-tp2-session-trace-replay-2026-08-18`
- Reason: add one implementation-neutral session trace bundle that can be recorded by the Demo API or Canonical offline runner, then replayed independently and in parallel by Canonical, Demo single-card, or Demo TP2 targets with selectable teacher-forcing stages.

## Scope

- Record input/session metadata and LLM, TTS, and token2wav token lineage in one immutable bundle.
- Reuse the same tracing and replay controls in API and offline runners.
- Support independent forcing of LLM tokens, TTS conditions, TTS acoustic tokens, and vocoder state.
- Keep Canonical and Demo replay outputs in separate directories so jobs can run concurrently.
- Validate the workflow with a short end-to-end case.
