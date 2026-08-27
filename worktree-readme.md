# Worktree

- Created: 2026-08-26
- Base branch: `align-enhance-fc-speedup-full-bundle-2026-08-26`
- Base commit: `8b890774f0a18862b3c625ed45ded93d76766d32`
- Branch: `align-enhance-fc-speedup-full-bundle-2026-08-26-docker-2026-08-26`
- Purpose: prepare and validate the Docker deployment path for the FC + TP2
  full-bundle implementation. The 34 machine is used for venv/model setup,
  bare-metal validation, image builds, and container validation.

## Docker validation (2026-08-26)

- The remote Docker host is `weihongliang@34.66.244.63`; it does not share the
  local filesystem. The worktree was synchronized there with `rsync`.
- Remote worktree:
  `/home/weihongliang/MiniCPM-o-Demo-wt-align-enhance-fc-speedup-full-bundle-2026-08-26-docker-2026-08-26`
- Runtime venv used for the remote client and bare-metal checks:
  `/home/weihongliang/MiniCPM-o-Demo-wt-align-enhance-fc-speedup-full-bundle-2026-08-26-docker-2026-08-26/.venv-high-cu128`
- Safetensors bundle mounted read-only at runtime:
  `/home/weihongliang/o5-artifacts/job672317_iter4000/o5_full_hf_fc_job672317_iter4000_public_llm_20260826`
- Processor and Token2Wav assets mounted read-only at runtime:
  `/home/weihongliang/o5-artifacts/assets/MiniCPM-o-4_5-assets`
- Worker image: `minicpm-o5-full-worker:docker-20260826`.
  Gateway image: `minicpm-o5-full-gateway:docker-20260826`.
- The worker image includes `gcc` and `libc6-dev` because FLA/Triton compiles
  a CUDA helper on first use. `gcc` alone was insufficient because
  `stdlib.h` was absent from the slim base image.
- The worker/backend bundle was run only on the free remote GPU1. The existing
  GPU0 processes and containers were not stopped or modified.
- The independent test gateway was exposed as `https://127.0.0.1:18010` on
  the remote host. It registered one idle worker and passed the video realtime
  probe using `o5-artifacts/assets/MiniCPM-o-4_5-assets/omni_duplex1.mp4`.
- Probe session: `sess_8db149d90b37`; result:
  `run-logs/docker-smoke-20260826/video_probe.json` on the remote host.
  The probe reported 11 text delta chunks and 10 output audio chunks.

## Selectable Docker topology

- `docker-compose.deploy.yml` now exposes two explicit Compose profiles:
  `--profile single` binds one GPU and starts `single_eager`; `--profile tp2`
  binds `TP2_GPU0` and `TP2_GPU1` to one container and starts
  `core/deploy/launch_tp2.sh` with two `torchrun` ranks.
- The worker entrypoint validates `O5_DEPLOY_MODE` and selects the backend
  launcher accordingly. `SINGLE_DEPLOY_MODE=single_eager` keeps the plain
  single-card path; `SINGLE_DEPLOY_MODE=single_opt` enables the single-card
  optimization engine. TP2 receives the acceleration flags through Compose,
  with LLM Graph, TTS fast, TTS Graph, batched MM, vision batching, and fused
  vision/audio enabled by default; vocoder graph remains opt-in.
- The selectable topology change requires rebuilding the worker image because
  the entrypoint is part of the image. The dependency layers remain reusable;
  model weights and runtime assets remain read-only runtime mounts.
