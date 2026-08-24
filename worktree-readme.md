# Worktree

- Created: 2026-08-24
- Base branch: `wt/o5-no-fc-speedup-tp2-default-local-load-2026-08-22-clean-replay-2026-08-23`
- Base commit: `a50b0cf` (`docs: record clean replay worktree`)
- Branch: `wt/o5-no-fc-speedup-tp2-default-local-load-2026-08-22-clean-replay-2026-08-23-strategy-grouped-unit-prefill-2026-08-24`
- Purpose: integrate the validated strategy-HD slice scheduler, dynamic grouped-MoE prefill, and one-shot multimodal unit prefill on top of the clean local-load/replay baseline.
- Scope: preserve the local-load behavior; cherry-pick only the required implementation and verification code from the strategy-HD worktree, without merging its unrelated deployment, server, reports, or experimental history.
