# Worktree: wt/o5-no-fc-speedup-tp2-default-local-load-2026-08-22

- Created: 2026-08-22 UTC
- Base branch: `o5-no-fc-speedup-tp2`
- Base commit: `ad070af` (`feat(replay): capture T2W intermediates and token sensitivity`)
- Purpose: make Demo loading self-contained by default. Replace the runtime
  dependency on the legacy `.pt` checkpoint with one complete safetensors
  bundle, keep an optional assets directory, and validate default loading and
  Canonical alignment without requiring callers to pass `model-path`.
