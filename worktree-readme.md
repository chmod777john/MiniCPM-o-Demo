# Docker environment from fixed installable wheels

Created 2026-10-14 UTC from `o5-fc-dev-external-hf-20261013-hf-chat-20261014` at `8ba58be08ef71988d490ad538be4b3734a0f5119`.
Owner: Codex subagent docker_lis_package. Purpose: replace installed site-packages archive with genuine pip wheel installation and auditable provenance.
Scope: Docker/runtime packaging, fixed wheel/source input manifests, install/import/native/API evidence and docs. HF/accel numerical sources unchanged. Root formal worktree and prior packaging tree are read-only.
Evidence: `/user/weihongliang/analysis-runs/20261014-docker-pip-install/`.
Protected: all unrelated host34 services and existing model/native cache. Root later stopped its browser pair and transferred GPU0/1 for coordinated acceptance. User cancelled the tmpfs engine; normal Docker build awaits storage/host handoff. No new image push or LIS writes.
Deliver through explicit reviewed commits to root.
