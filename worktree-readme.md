# Worktree: LIS 828 packaging

- Created: 2026-08-28
- Base branch: `align-enhance-fc-speedup-full-bundle-2026-08-26-docker-2026-08-26`
- Base commit: `303bbf3 docs(docker): add FC deployment and probe guide`
- Branch: `align-enhance-fc-speedup-full-bundle-2026-08-26-docker-2026-08-26-lis828`
- Purpose: adapt the two-image O5 FC deployment for LIS, where the worker receives
  one mounted model artifact and fixed runtime defaults instead of the Compose
  host-path layout and per-run environment overrides.
- Scope: keep the existing Compose deployment behavior intact; LIS does not need
  session recording for this validation.
