# External-only Demo adapter

Created 2026-10-19 UTC by Codex/root from demo-adapter-fc@b1ac65015e9283f6bb9482ee25463dd98ad3ca3e.
User authorized removing vendored O5/O45 model implementations, old fallback loading/acceleration paths and dependent entries.
Scope: this Demo worktree; preserve external HF/accel model semantics, FC/API/application behavior.
Avoid: live frozen 9007/9009 deployments, shared environments, HF/accel source. No GPU task allocated.
Completion: coherent external-only source tree, static dependency review, committed change; fast-forward formal demo-adapter-fc.
No new tests or inference runs requested for this cleanup. Existing tests that address deleted implementations must be removed or adapted.
