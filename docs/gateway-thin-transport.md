# O5 Gateway thin transport contract

Status: candidate branch `gateway/thin-realtime-20261103`.

## Boundary

The Gateway owns four things:

1. model discovery (`GET /v1/models`);
2. Worker registration, health, draining and FIFO scheduling;
3. session watchdog and retained-WebSocket reconnect;
4. transparent forwarding between the client and one selected Worker.

It does not serve HTML, frontend assets or the documentation site. Those belong
to the separate application service (`demo_server.py`).

## Public entry points

- `WS /v1/realtime?model=<model-id>&mode=chat|audio|video|omni`
- `WS /e/<endpoint>/v1/realtime?mode=audio|omni`
- `GET /v1/models`
- `GET /e/<endpoint>/v1/info`

Both WebSocket entry points converge on `_realtime_entry()`. `mode` remains
application protocol metadata; the Gateway uses it only to choose a scheduling
request type and capacity policy.

## Gateway to Worker

Every mode uses exactly one internal route:

```text
/v1/worker/realtime
```

The original client query parameters are forwarded unchanged, with Gateway
identity metadata added when absent. Text and binary protocol frames are sent
without inspecting, renaming or rewriting their JSON fields.

The Gateway may emit its own scheduling frames before the Worker session starts:

- `session.queued`
- `session.queue_update`
- `session.queue_done`
- `error`
- Gateway-originated `session.closed` on timeout or forced drain

Everything produced by the Worker is forwarded verbatim. Application APIs,
presets, reference-audio storage and session recording are outside this service.

## Worker lifecycle assumptions

- A Worker accepts at most one active session at a time.
- `/health` reports loading/idle/busy/duplex state and capabilities.
- A disconnected Worker is marked offline and the assigned connection retries
  dispatch once before failing the session.
- Draining stops new assignments but does not interrupt the current session.
- Worker failure ends its current session; this contract does not migrate KV or
  duplex state.

## Verification

The CPU regression set is:

```bash
python -m pytest -q   tests/test_gateway_endpoints.py   tests/test_gateway_api_recording.py   tests/test_fc_api_v5_input_mode.py   tests/test_upstream_refresh_routing.py
```

It covers named endpoints, queueing, unreachable Worker recovery, force close,
frame transparency and retained reconnect.
