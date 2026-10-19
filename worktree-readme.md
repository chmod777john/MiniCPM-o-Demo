# Minimal simplex Demo follow-up

- Created: 2026-10-14 UTC.
- Owner: Codex subagent `/root/hf_minimal_simplex_rewrite`.
- Base branch: `o5-fc-dev-external-hf-20261013-hf-chat-20261014`.
- Base commit: `d036b7a6`.
- Candidate branch: `o5-fc-dev-external-hf-20261013-hf-chat-20261014-minimal-simplex`.
- Purpose: pair with minimal HF simplex; remove simplex length_penalty exposure and bridge; connect desktop cancellation to the current realtime API and isolate late replies.
- Scope: simplex page/parameter/view/backend files; no ordinary/FC duplex algorithms or shared video/share files owned by another agent.
- Evidence: `/user/weihongliang/analysis-runs/20261014-hf-simplex-minimal/`.
- Validation: authorized CPU and agent-dev GPU/API probes; exact results recorded separately. Fixed packaged images are not changed.


## Demo upstream refresh candidate — 2026-10-19T08:08:22.751801+00:00

- Owner: `/root/demo_refresh`.
- Base branch: `demo-adapter-fc`.
- Base commit: `e760421992b36459fb3e724be38d7caa9cb83551`.
- Candidate branch: `demo-adapter-fc-upstream-refresh-20261019`.
- Purpose: merge pinned `o5-fc-dev@e0f2ada084d395a00d1956957d345e82b8a5f8f7` while retaining external HF/accel and ordinary/simplex/mobile/deployment features.
- Scope: this worktree only; no formal branch update, remote push, service changes, builds, or tests.
- Handoff: candidate merge commit plus `/user/weihongliang/analysis-runs/20261019-o5fc-upstream-refresh/demo/REPORT.md`.
- Earlier provenance in this file is retained as history.
