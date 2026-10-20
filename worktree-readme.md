# External-only Demo adapter

Created 2026-10-19 UTC by Codex/root from demo-adapter-fc@b1ac65015e9283f6bb9482ee25463dd98ad3ca3e.
User authorized removing vendored O5/O45 model implementations, old fallback loading/acceleration paths and dependent entries.
Scope: this Demo worktree; preserve external HF/accel model semantics, FC/API/application behavior.
Avoid: live frozen 9007/9009 deployments, shared environments, HF/accel source. No GPU task allocated.
Completion: coherent external-only source tree, static dependency review, committed change; fast-forward formal demo-adapter-fc.
No new tests or inference runs requested for this cleanup. Existing tests that address deleted implementations must be removed or adapted.

## Formal branch integration 2026-10-20T06:26:39.776254+00:00

Owner: root. Existing worktree reused; checked out formal `demo-adapter-fc` from `c4a72a68d078b32fbe33ae60c160294af7d72100`.
User authorized merging the native simplex OpenAI API and representative benchmark records from `demo-adapter-fc-openai-simplex-20261020-multibench@4c1ec09cb90a0880c5927dd26643656cee167fac`.
The validated runtime files are retained unchanged. HF, accel, MAP application branch and live deployments are outside this merge. Existing benchmark evidence is reused; no new GPU job or test run.
