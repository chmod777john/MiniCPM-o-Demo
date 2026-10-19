# Demo refresh CPU and frontend validation — 2026-10-19

## Versions and scope

- Demo product source: `ba64bb95f70a118b435ad336ad2fa4eec3d13258`.
- Corresponding HF: `db7541095e458bf71269a49fafea508369556574`; accel: `337ac3e84b69dde135efaad7f4113bc244fe7bd4`. Those were not executed as real model inference in this CPU task.
- Worktree: `/user/weihongliang/worktrees/demo-adapter-fc-upstream-refresh-20261019`.
- No product code changes. Added `tests/test_upstream_refresh_routing.py` with 20 focused regressions.
- No GPU allocation, public services, image builds, remote pushes or formal branch updates. Existing gateway tests use short-lived loopback fake workers.

## Final results

| Run | Result | Scope |
|---|---|---|
| `protocol-gallery-01` | 294 passed | v5/v4 mode rules, gateway endpoints, gallery, uicard, control-plane, protocol/media/recording helpers |
| `fc-runtime-o5-02` | 144 passed, 4 explicitly deselected | O5 FC/runtime/DTO/adapter/resume/component tests; excludes four documented legacy O45 import failures |
| `additional-o5-01` | 184 passed | FSM, constrained decode, standalone event ordering, wrappers, checkpoint utilities, tools/gallery and launch contracts |
| `refresh-boundaries-02` | 20 passed | New merge boundary regressions including real loopback gateway retained transport reconnect |
| `combined-o5-01` | **642 passed, 4 deselected** | All the above in one pytest process; 32.36 seconds pytest time |
| `mobile-desktop-02` | **25 passed** | Four mobile contract files plus desktop chat cancellation, Node 22.23.3 |
| standalone Python scripts | **3 exit-0 passes** | External ChatView history/options/audio conversion, cancellation drain, external accel options |

Do not sum repeated suite runs into a larger test count. The 642 combined cases are unique across the selected Python suite. The three standalone Python contract files execute assertions during import and have no pytest test functions; explicit script execution provides their separate exit-code evidence.

## New regressions

- Named audio/omni mode is forwarded through direct and retained gateway paths.
- Ordinary reconnect keeps its unlimited default duration; ordinary video and new omni use the same timeout class otherwise; explicit environment overrides remain effective.
- Default ordinary audio/video entry leaves input_mode unset.
- Ordinary backend prepare does not enter FC merely due to a protocol_version field.
- O5 model adapter forwards Omni mode and frame_list; video seconds are projected to milliseconds and visual token count is retained.
- A real Gateway TestClient plus loopback fake worker receives the mode through retained transport, disconnects, reattaches with reconnect_token, and continues without reinitialization. This is transport/bridge validation, not model inference.

## Preserved failures and explanations

1. `fc-runtime-vendored-01`: 144 passed and four failed. All four are `test_o45_chain_*` in `tests/test_fc_duplex_spoken_kwargs_chain.py`. O45 imports call `AutoImageProcessor.register("MiniCPMVImageProcessor", ...)`; transformers 5.13.0 rejects the string config key with AttributeError. The four relevant O45 source files and the test file are byte-identical to premerge `e7604219` (see `legacy-o45-baseline.json`). No O45 code was changed. The O5 rerun explicitly names the four deselections rather than hiding the first failure. We did not run a separate old checkout: unchanged-source equivalence establishes that this code was not introduced by the merge.
2. `mobile-desktop-01`: 16 passed, one test-file import failed because the new worktree lacked TypeScript. Installed TypeScript 6.0.3 only in this worktree's `.test-tools` directory via npm; the mobile node_modules entry points there. Full rerun passes25/25. No shared environment changed.
3. `test_external_hf_chat_contract.log` contains expected logged ValueErrors when stale prepared chat input is intentionally reused; the script catches these and exits0.

## Environment

Python `/user/weihongliang/environments/o5-3drope-fc-20261013/.venv/bin/python`, read-only reuse. Python3.10.12, torch2.11.0+cu128, transformers5.13.0, SDK0.0.6, minicpmo-utils1.0.7, pytest9.1.1, FastAPI0.139.0. Full versions and resolved module origins are in `ENVIRONMENT.json`; some packages resolve to the historical high-cu128 base venv through the existing environment. Nothing was pip-installed.

Node `/user/weihongliang/.local/share/claude-code/node-v22.23.3/bin/node`; system Node16 was not used for these tests. TypeScript6.0.3 installed project-locally with `npm install --prefix .test-tools --no-save --ignore-scripts --package-lock=false typescript@6.0.3` and linked into this worktree's frontend/mobile/node_modules.

All Python test commands set CUDA_VISIBLE_DEVICES empty and OMP_NUM_THREADS=1. Per-run `.json` files contain exact argv/cwd/version/environment selection, `.log` contains complete output, `.xml` contains pytest results. `run_cpu_suite.py` is the evidence runner.

## Boundaries

- 8 TTS condition, 13 canvas positions and 8 bounded-prefill cases exercise **vendored upstream implementations**; they are not evidence that external HF/accel is numerically correct.
- The external chat bridge tests use the actual ChatView/API methods with fake model outputs. No checkpoint was loaded, no GPU audio/video generated, no quality/speed claim.
- Node tests execute real frontend helpers and selected request/cancellation code under mocks; no browser microphone/camera or visual UI acceptance, no frontend production build.
- Existing model-loading tests (`test_chat.py`, `test_streaming.py`, `test_duplex.py`, etc.) were not included in this CPU suite. Real external-HF inference is owned by the GPU agent.

## Handoff

Product changes: none. Candidate test+record commit may be fast-forwarded by root. No active test processes or services remain. Formal branch is not updated by this agent.
