# External HF Docker package

This image assembles three independent committed materials under `/opt/o5/{hf,accel,demo}`. Runtime imports external HF directly; acceleration is attached from the separate package. No checkpoint export or modeling copy into Demo is performed.

## Build input

`materials/hf.tar`, `materials/demo.tar`, `materials/accel-runtime.tar` are Git archives; `materials/mobile/` is the current Demo build:static output. The assembler requires clean committed inputs and runs `npm run build:static` with the selected Node/npm toolchain. `materials/source-manifest.json` records application commits, packaging commit and per-file/tree hashes, mobile build inputs/tools and mobile/mobile-omni output hashes. The latter covers generated React assets and the Omni shell/widget. The acceleration archive includes src, pyproject.toml and README only, excluding benchmark evidence.

Runtime packages are installed into a fresh `/opt/o5/.venv` from original or fixed-source-built wheels. See [pip/README.md](pip/README.md) for pinned inputs, native build provenance and the two explicit utils/librosa and original decord platform-tag metadata exceptions. `pip/public-wheel-inputs.json` contains official source URLs and SHA256; `requirements-install.lock` is generated only after every actual wheel has been verified. `requirements-pro5000.lock` remains a historical installed-version inventory, not the new installation lock.

`apt-offline/` must first contain the fixed original Debian packages and signed indexes described in `pip/README.md`; target build nodes need no external package network. `pip/download_locked.py` prepares `wheelhouse/`, official CUDA compiler `sources/`, and the public hash lock. Add the original SDK/utils wheels and the newly source-built FA2/causal/dense wheels, then run `pip/assemble_installation.py` with their native build manifest to produce the complete `requirements-install.lock` and `installation-manifest.json`. Final Docker build runs pip with `--no-index --no-deps --require-hashes`; no existing site-packages archive is restored. The wheel-provided Ninja executable is installed by pip. A CPU nvcc compilation checks the assembled official CUDA redistributions, and an installed dependency audit permits only the two documented metadata exceptions (utils/librosa and the fixed original decord WHEEL tag).

The default Dockerfile now describes this pip path. The pip image at packaging commit `79177c66` has completed its 37-node BuildKit construction, 137-distribution audit with two explicit metadata exceptions, and 14 container CPU gates. Its actual gateway process also serves desktop/mobile HTTP pages and exits cleanly. The same fixed new images subsequently passed real Pro5000 TP2 native A/B (17 exact outputs), 12 simplex requests, ordinary audio/video duplex, an 8-unit FC fixture and cross-mode recovery. Scope and timing limits are recorded in the acceptance report; historical image results later in this document still apply only to the older snapshot images. See [PIP_IMAGE_ACCEPTANCE.md](PIP_IMAGE_ACCEPTANCE.md).

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

The fixed NVIDIA cuBLAS wheel contains versioned `.so.13` files. The image provides link-time `libcublas.so`/`libcublasLt.so` symlinks under CUDA_HOME/lib64 for acceleration C++ extension builds; both links target the unchanged runtime libraries. The native single smoke does not exercise this grouped-extension linker path.

The local v9 image above passed TP2 API probes on two Pro6000 using the Pro5000 acceleration preset. It has not been pushed. The previously published v6 tag predates the TP2 compatibility repairs and is not the TP2 delivery image. See `docs/external-hf/DOCKER_TP2_ACCEPTANCE.md` for scope and restart commands.

## Historical LIS split container contract

`Dockerfile.split-overlay` changes packaging only on the accepted TP2 image;
`Dockerfile.gateway-role` fixes its companion image to gateway. No runtime package
or HF/accel source is replaced. `lis-contract.json` describes the selected connection
contract (a fragment for template configuration, not a complete LIS resource).
The worker image fixes `periodic_idempotent`, refresh 20s and max backoff 20s. The
initial failed PUT retries after 1/2/4/8/16/20s; HTTP timeout is 5s. A compatible
template must declare those same values. This choice supports gateway restart;
`startup_retry_only` is a different historical contract, not this release.

Gateway forwards LIS CLI arguments, including nondefault ports and `--http`.
It reads `/app/config.json` without modifying it, preserving recording and all
supplied fields in `/tmp/o5-entrypoint/config.json`. Missing data-dir defaults to
`/app/data` when a platform config is supplied. The optional historical
`/etc/lis/publisher-runtime/runtime.json` provides fallback gateway ports and is
checked for registration-contract mismatch. Historical worker Pods do not mount
that ConfigMap, so registration defaults are baked into the image.

Worker reads the template's `MODEL_PATH`, `WORKER_ID`, `WORKER_ENDPOINT`,
`WORKER_GPU_GROUP`, `WORKER_PORT`, `BACKEND_PORT`, and `GATEWAY_REGISTRY_URL`.
The model root contains `model.pt` and `token2wav_causal_vocoder/`. Backend binds
`0.0.0.0`; the colocated worker connects over `127.0.0.1`. Exactly one worker Pod
needs two GPUs for the fixed TP2 preset. Gateway has no GPU requirement.

Registration uses the existing gateway's idempotent PUT. The packaging supervisor
checks children between bounded requests; it has no detached registrar thread or
process. Model loading is independent of registry availability. Restarting the
gateway can terminate active client sessions; registration recovery enables new
sessions, not transparent replay of an interrupted conversation.

CPU checks (no model imports):

```sh
python docker/external-hf/test_entrypoint.py
python docker/external-hf/test_lis_contract.py
python docker/external-hf/test_launcher_image.py
```

For historical `lisctl template base create-realtime`, use the **flat**
`lis-contract-flat.json` as `--contract-json`; the nested `lis-contract.json` is
only a description of runtime fields. `lis-resources-pro5000.example.json` is a
conservative planning example (16 CPU, 192 GiB host RAM, two GPUs), not a measured
minimum or physical Pro5000 admission guarantee. Replace its `gpu_types` string
with the cluster's actual registered SKU. Two Pro6000 GPUs on host34 provide the
local acceptance environment. A final LIS command, after separately publishing
and approving actual image records, has this shape:

```sh
lisctl template base create-realtime \
  --name o5-external-hf-tp2 \
  --release-id o5-external-hf-20261014-split \
  --gateway-image-id '<LIS gateway image UUID>' \
  --worker-image-id '<LIS worker image UUID>' \
  --contract-json @docker/external-hf/lis-contract-flat.json \
  --resources-json @docker/external-hf/lis-resources-pro5000.example.json
```

Image UUIDs above are platform records, not Docker image IDs. This command is a
handoff recipe; this task does not execute it or create platform resources.
