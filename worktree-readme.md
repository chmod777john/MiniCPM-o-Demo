# Worktree

- Created: 2026-08-26
- Branch: `align-enhance-fc-speedup-full-bundle-2026-08-26`
- Base branch: `align-enhance-fc`
- Base commit: `5039266`
- Purpose: merge the relevant `o5-no-fc-speedup-tp2` complete safetensors/no-modelpath and grouped-MM history into the FC integration without changing the existing FC worktree.
- Scope: preserve the FC/LLM Graph/TP2 architecture, add the speedup artifact-loading and long-prefill changes, then validate the resulting FC replay path.
