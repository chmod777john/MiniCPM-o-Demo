# Worktree

- Created: 2026-08-23
- Base branch: `o5-no-fc-speedup-tp2`
- Base commit: `261dcea` (`fix(tokenizer): remove stale added-token override`)
- Branch: `wt/o5-no-fc-speedup-tp2-default-local-load-2026-08-22-clean-replay-2026-08-23`
- Purpose: isolate the clean default local-load result from the later uncommitted diagnosis/ablation probes, then rerun the LLM teacher-forcing replay and argmax-reversal check.
- Scope: retain the complete safetensors artifact path and default local assets; do not carry uncommitted experimental switches or probe code from the source worktree.
