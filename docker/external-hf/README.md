# External HF Docker package

This image assembles three independent committed materials under `/opt/o5/{hf,accel,demo}`. Runtime imports external HF directly; acceleration is attached from the separate package. No checkpoint export or modeling copy into Demo is performed.

## Build input

`materials/hf.tar`, `materials/demo.tar`, `materials/accel-runtime.tar` are Git archives; `materials/mobile/` is the current Demo build:static output. The assembler requires clean committed inputs and runs `npm run build:static` with the selected Node/npm toolchain. `materials/source-manifest.json` records application commits, packaging commit and per-file/tree hashes, mobile build inputs/tools and mobile/mobile-omni output hashes. The latter covers generated React assets and the Omni shell/widget. The acceleration archive includes src, pyproject.toml and README only, excluding benchmark evidence.

`runtime/pro5000-python310-site.tar.gz` contains the verified Pro5000 project's site-packages and its qualified native dense causal artifact. `tools/ninja` must be copied from the source venv/bin (the installed Python ninja package does not embed that executable). `runtime/cuda13-compiler.tar.gz` contains nvcc, nvvm, CUDA13 headers and libcudart. It is intentional to restore this verified Python3.10/Torch2.14+cu130 environment: the older requirements-pro5000.txt described a discarded Python3.12/Torch2.9 candidate. Installed versions are captured in requirements-pro5000.lock. Source environment provenance and archive SHA256 belong in the delivery manifest.

SDK0.0.6 and minicpmo-utils1.0.7 wheels go in wheels/. They are installed without dependency resolution, preserving the tested runtime. minicpmo-utils optional metadata has older Torch/Transformers/librosa pins; API tests, not pip's old optional pins, qualify this exact combination.

Build from assembled context:

```sh
sudo docker build -f docker/external-hf/Dockerfile -t o5-external-hf:20261014 .
```

No global pip install occurs: all Python dependencies are in `/opt/o5/.venv`. The container includes compiler headers because some acceleration extensions compile on first use.

## Model artifact

Mount one directory as MODEL_PATH with:

```text
model.pt                       # original raw checkpoint
token2wav_causal_vocoder/       # flow.pt, hift.pt, ONNX/config assets
```

For Docker standalone TP2 on two free Pro5000 GPUs:

```sh
sudo docker run --rm --gpus '"device=0,1"' --ipc=host \
  -e O5_ROLE=standalone -p 18014:8006 \
  -v /path/to/model:/models/o5:ro o5-external-hf:20261014
```

Only use unoccupied devices. GPU model and precision/performance scope must be recorded separately. The current host34 has Pro6000; a Pro6000 result is not Pro5000 hardware acceptance. For a native single-GPU functional check override `O5_DEPLOY_MODE=single_eager`; no Pro5000 single acceleration preset is qualified.

The default worker role starts backend+worker and can register to GATEWAY_REGISTRY_URL. `O5_ROLE=gateway` starts the gateway without loading weights. `O5_ROLE=standalone` includes all three for local API validation. New gateway/worker assets may share a source image initially; a smaller gateway image is an optional packaging optimization.

## LIS

MODEL_PATH is parsed at runtime, matching the prior single ModelWeight mount convention. Acceleration and HF paths/defaults are baked into image ENV so custom platform env injection is unnecessary. Prior August platform limitations are historical evidence, not a current permission guarantee. Inspect current lisctl/template contract before registration; do not overwrite active production assets.

Current task's `lisctl auth whoami` returned401; image API validation continues independently. No LIS deployment or Pro5000 TP2 acceptance is claimed until it actually runs.

A LIS gateway asset must have its role fixed in the image when custom environment injection is unavailable. Build the thin `Dockerfile.gateway-role` with `WORKER_IMAGE=<qualified worker image>`; it sets `O5_ROLE=gateway`. Do not register the default worker-role image as a gateway without this layer or an explicit supported template override.

`Dockerfile.overlay` is for updating packaging files and ninja atop a qualified immutable base image on disk-limited build hosts. Pin the base digest in the release manifest; source/runtime archives and the full Dockerfile remain the rebuild recipe.


The current default reference audio is Demo's archived `assets/ref_audio/ref_minicpm_signature.wav`, resolved by `cfg.audio.ref_audio_path`. No ineffective `REF_AUDIO_PATH` environment override is set. Explicit API reference audio follows the request path.

A final delivery manifest must supplement source archives with the runtime snapshot/native/ninja/checkpoint hashes, immutable base and final worker/gateway image IDs and registry digests. For a thin overlay, `release-input-manifest.json` records its exact package and inherited inputs; final image digests live in the external delivery manifest to avoid a self-referential digest.

Both full and overlay builds replace the generated mobile/mobile-omni trees completely; copying into an old tree can retain obsolete content-hashed JavaScript. Overlay context must include freshly built `static/mobile` and `static/mobile-omni`, matching the input manifest hashes.

## Pending TP2 acceptance

`start_tp2.py` is a prepared launcher. It checks both explicitly selected GPUs for compute processes and memory occupancy before invoking the container. It never stops workloads. It requires a complete local image ID or immutable image digest and sets standalone API role explicitly. This launcher is not a claim that the current image has passed TP2 API acceptance.

For Docker on a host with two owner-authorized free GPUs:

```sh
python3 docker/external-hf/start_tp2.py --engine docker --sudo \
  --devices 0,1 --model-dir /path/to/model --logs-dir /path/to/logs \
  --image sha256:93aa6ee9df9abc96669d09235bcbd499be6a369c4a4fa63ec3507e3402afaaf2
```

For the Pro5000 host, `docker` is a rootless Podman compatibility wrapper. Use `--engine podman` without `--sudo`, and set `CONTAINERS_CONF` to the task-local configuration pointing at generated NVIDIA CDI metadata. No global runtime configuration changes are required by this package. Device injection and actual GPU startup remain pending until two GPUs are free.

`Dockerfile.app-overlay` refreshes explicit committed application paths atop the immutable accepted image. The release manifest must enumerate the inherited application source, patch source/paths and actual file hashes; never relabel old native API evidence as a run of the new application layer.

The qualified runtime snapshot contains versioned cuBLAS `.so.13` files. The image provides link-time `libcublas.so`/`libcublasLt.so` symlinks under CUDA_HOME/lib64 for acceleration C++ extension builds; both links target the unchanged runtime libraries. The native single smoke does not exercise this grouped-extension linker path.

The local v9 image above passed TP2 API probes on two Pro6000 using the Pro5000 acceleration preset. It has not been pushed. The previously published v6 tag predates the TP2 compatibility repairs and is not the TP2 delivery image. See `docs/external-hf/DOCKER_TP2_ACCEPTANCE.md` for scope and restart commands.
