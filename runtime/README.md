# Runtime Layer

This package is the worker-local boundary between transport handlers and model
backends.

## Current Shape

The worker exposes one realtime endpoint for turn-based chat and duplex requests:

- `/v1/realtime`：Worker 根据 `session.init.payload.mode` / `input_mode` / query `mode` 内部选择 turn-based 或 full-duplex。

Public clients may enter through the Gateway at `/v1/realtime`. After Worker
assignment, the Gateway forwards realtime frames without selecting by mode or
translating the payload. The Worker resolves the session mode internally.

Inside the worker, duplex sessions flow through:

```text
worker.py WebSocket handler
  -> RuntimeManager
  -> DuplexSessionRuntime
  -> DuplexBackendAdapter
  -> PyTorch model methods (or future C++/SGLang/vLLM adapters)
```

Gateway worker scheduling now also has a small capability contract.  Worker
health responses report capabilities such as `chat`, `streaming`,
`audio_duplex`, and `omni_duplex`; `WorkerPool` keeps these on
`WorkerConnection` and only assigns a request to an idle worker that advertises
the required capability.  Existing workers default to all capabilities, so this
is a compatibility-preserving boundary for future specialized runtimes.

## Responsibilities

### worker.py

- Host the unified Worker realtime WebSocket endpoint.
- Parse worker runtime protocol messages.
- Own worker process state and expose health/status endpoints.
- Delegate inference lifecycle to runtime/backend boundaries.

### RuntimeManager

- Own worker-local runtime instances by session id.
- Close runtimes on session end or worker shutdown.

### DuplexSessionRuntime

- Own per-session inference lifecycle.
- Convert one input frame into backend prefill/generate work.
- Manage deferred finalize internally.
- Drain finalize before stop/cleanup.

### BackendAdapter

- Hide backend-specific execution mechanics.
- PyTorch may need explicit finalize.
- C++ may treat finalize as a no-op and produce audio asynchronously.
- Future backends should satisfy the same adapter contract.

## Design Rule

`finalize`, KV-cache edits, prompt-cache artifacts, and backend-specific stream
mechanics should not leak into gateway or scheduler code.  They belong inside
runtime/backend boundaries.
