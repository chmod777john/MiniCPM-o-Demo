# External HF Docker package

This image assembles three independent committed materials under `/opt/o5/{hf,accel,demo}`. Runtime imports external HF directly; acceleration is attached from the separate package. No checkpoint export or modeling copy into Demo is performed.

## Build input

`materials/hf.tar`, `materials/demo.tar`, `materials/accel-runtime.tar` are Git archives; `materials/mobile/` is the current Demo build:static output. `materials/source-manifest.json` records their commits. The acceleration archive includes src, pyproject.toml and README only, excluding benchmark evidence.

`runtime/pro5000-python310-site.tar.gz` contains the verified Pro5000 project's site-packages and its qualified native dense causal artifact. `runtime/cuda13-compiler.tar.gz` contains nvcc, nvvm, CUDA13 headers and libcudart. It is intentional to restore this verified Python3.10/Torch2.14+cu130 environment: the older requirements-pro5000.txt described a discarded Python3.12/Torch2.9 candidate. Installed versions are captured in requirements-pro5000.lock. Source environment provenance and archive SHA256 belong in the delivery manifest.

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
