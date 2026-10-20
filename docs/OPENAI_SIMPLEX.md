# Native HF simplex benchmark API

This small compatibility service calls the existing `mode=chat` endpoint, which invokes external HF `MiniCPMO.chat`. It preserves the supplied history and images. It does not use FC units, add prompts, retry empty answers, or modify HF/accel.

## Start

Run the matching Demo backend/worker/gateway first, with the external HF and accel paths configured as usual. Then, using a project virtual environment with FastAPI, uvicorn, websockets>=14 and Pillow:

```sh
O5_CHAT_WS_URL='ws://127.0.0.1:8006/v1/realtime?mode=chat' \
O5_MODEL_NAME=minicpm-o5 PORT=18080 \
.venv/bin/python openai_app.py
```

The default HTTP bind address is127.0.0.1. Use `HOST` to select another address. `ADAPTER_API_KEY` enables inbound Bearer authentication; unset means no authentication for local/private deployments. `O5_CHAT_API_KEY` optionally authenticates the upstream WebSocket. `O5_CHAT_TIMEOUT` defaults to600seconds. Health reports application liveness, not upstream model availability. The existing Gateway retains queueing and per-worker serialization.

## Contract

- `GET /health`; `GET /v1/models`; `POST /v1/chat/completions`.
- Model name defaults to `minicpm-o5`.
- Complete `system/user/assistant` history, ending in a user message.
- Text strings or text/image_url content arrays; inline base64 PNG/JPEG images only. Multiple images retain order. `detail=high/auto` uses the native HF image processor; this does not emulate OpenAI image token budgets. `detail=low` is rejected.
- Nonstreaming text output only (`stream=false`, `n=1`). Audio endpoints, video, tool calls, strict JSON, nonempty stop sequences, seed, logprobs and nonzero frequency/presence penalties are not implemented and are rejected.
- `max_tokens` or `max_completion_tokens`,1–4096; default2048. Conflicting aliases are rejected.
- `temperature=0` selects single-beam greedy generation. Positive temperature and top_p are forwarded. Omitted sampling values use HF defaults0.7/0.8. Other native defaults, including repetition penalty, remain HF defaults.
- Response includes actual `stop`/`length` and LLM token usage observed from the native HF processor/generation output. Image placeholders expanded by HF count as LLM input positions; this is not OpenAI billing token accounting. Old backends without completion metadata fail explicitly rather than fabricating usage.
- Each HTTP call owns one turn-based WS session; closing it releases through the existing backend lifecycle. Cancelling a nonstreaming request may still require waiting for in-flight inference to finish safely.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="minicpm-o5", messages=[{"role":"user", "content":"What is 2+2?"}],
    temperature=0.0, max_tokens=2048,
)
print(reply.choices[0].message.content)
```

For VLMEvalKit use `VLLMOpenAIWrapper` with this base_url, model, timeout600, temperature0 and max_tokens2048. Keep its dataset prompt and exact_matching judge unchanged. API compatibility tests and a small AI2D sample demonstrate integration only, not a full benchmark score or comparability with the FC-based adapter.
