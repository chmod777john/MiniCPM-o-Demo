"""Small Chat Completions bridge over the native HF turn-based WebSocket API."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import json
import math
import os
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus


class APIError(Exception):
    def __init__(self, message, *, param=None, status=400, code="invalid_request_error"):
        self.message, self.param, self.status, self.code = message, param, status, code


@dataclass(frozen=True)
class Settings:
    upstream: str = "ws://127.0.0.1:8006/v1/realtime?mode=chat"
    model: str = "minicpm-o5"
    api_key: str = ""
    upstream_key: str = ""
    timeout_s: float = 600
    max_body_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_env(cls):
        return cls(upstream=os.getenv("O5_CHAT_WS_URL", cls.upstream),
                   model=os.getenv("O5_MODEL_NAME", cls.model),
                   api_key=os.getenv("ADAPTER_API_KEY", ""),
                   upstream_key=os.getenv("O5_CHAT_API_KEY", ""),
                   timeout_s=float(os.getenv("O5_CHAT_TIMEOUT", "600")))


def image_data(part, param):
    spec = part.get("image_url")
    if not isinstance(spec, dict) or set(spec) - {"url", "detail"}:
        raise APIError("image_url must contain url and optional detail", param=param)
    if spec.get("detail", "auto") not in ("auto", "high"):
        raise APIError("Only native-resolution detail=auto/high is supported", param=param)
    url = spec.get("url")
    if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
        raise APIError("Images must be base64 PNG/JPEG data URLs; remote/local paths are unsupported", param=param)
    encoded = url.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("unsupported image format")
            image.verify()
    except (ValueError, binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise APIError("Invalid PNG/JPEG image data", param=param) from exc
    return {"type": "image", "data": encoded}


def translate(body, model):
    if not isinstance(body, dict):
        raise APIError("Request body must be a JSON object")
    allowed = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p",
               "stream", "n", "stop", "presence_penalty", "frequency_penalty", "logprobs",
               "response_format", "tools", "tool_choice", "user"}
    for name, value in body.items():
        if name not in allowed and value is not None:
            raise APIError(f"Unsupported parameter: {name}", param=name)
    if body.get("model") != model:
        raise APIError(f"Unknown model; use {model}", param="model", status=404, code="model_not_found")
    if body.get("stream") not in (None, False):
        raise APIError("This benchmark endpoint supports stream=false only", param="stream")
    if body.get("n") is not None and (type(body["n"]) is not int or body["n"] != 1):
        raise APIError("Only n=1 is supported", param="n")
    for name in ("presence_penalty", "frequency_penalty"):
        if body.get(name) not in (None, 0):
            raise APIError(f"{name} is unsupported", param=name)
    for name in ("stop", "logprobs", "tools", "tool_choice"):
        if body.get(name) not in (None, False, []):
            raise APIError(f"{name} is unsupported", param=name)
    if body.get("response_format") not in (None, {"type": "text"}):
        raise APIError("Only response_format.type=text is supported", param="response_format")
    limit = body.get("max_completion_tokens")
    legacy = body.get("max_tokens")
    if limit is not None and legacy is not None and limit != legacy:
        raise APIError("Conflicting max_tokens and max_completion_tokens", param="max_tokens")
    limit = limit if limit is not None else legacy if legacy is not None else 2048
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise APIError("Token limit must be an integer between 1 and 4096", param="max_tokens")
    generation = {"max_new_tokens": limit}
    for name, default, minimum, maximum in (("temperature", 0.7, 0, 2), ("top_p", 0.8, 0, 1)):
        value = body.get(name)
        value = default if value is None else value
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not minimum <= value <= maximum or (name == "top_p" and value == 0)):
            raise APIError(f"Invalid {name}", param=name)
        generation[name] = value
    generation["do_sample"] = generation["temperature"] != 0
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise APIError("messages must be a nonempty list", param="messages")
    messages = []
    for index, message in enumerate(raw_messages):
        param = f"messages[{index}]"
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise APIError("Messages require role/content only", param=param)
        if message.get("role") not in ("system", "user", "assistant"):
            raise APIError("Supported roles: system, user, assistant", param=param + ".role")
        content = message.get("content")
        if isinstance(content, str):
            converted = content
        elif isinstance(content, list) and content:
            converted = []
            for part in content:
                if not isinstance(part, dict):
                    raise APIError("Content parts must be objects", param=param + ".content")
                if part.get("type") == "text" and set(part) <= {"type", "text"} and isinstance(part.get("text"), str):
                    converted.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and set(part) <= {"type", "image_url"}:
                    converted.append(image_data(part, param + ".content"))
                else:
                    raise APIError("Supported content: text and image_url", param=param + ".content")
        else:
            raise APIError("content must be text or a nonempty content list", param=param + ".content")
        messages.append({"role": message["role"], "content": converted})
    if messages[0]["role"] not in {"system", "user"} or messages[-1]["role"] != "user":
        raise APIError("History must begin with system/user and end with user", param="messages")
    return {"messages": messages, "streaming": False, "generation": generation,
            "tts": {"enabled": False}, "use_tts_template": False, "enable_thinking": False,
            "return_completion_metadata": True}


async def run_chat(payload, settings):
    headers = {"Authorization": "Bearer " + settings.upstream_key} if settings.upstream_key else None
    initialized = submitted = False
    completed = None
    async with connect(settings.upstream, additional_headers=headers, compression=None,
                       open_timeout=30, close_timeout=5, max_size=settings.max_body_bytes) as ws:
        async for raw in ws:
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "session.queue_done" and not initialized:
                initialized = True
                await ws.send(json.dumps({"type": "session.init", "payload": {}}))
            elif kind == "session.created" and not submitted:
                submitted = True
                await ws.send(json.dumps({"type": "input.append", "input": payload}))
            elif kind == "response.done":
                metadata = event.get("completion_metadata") or {}
                if metadata.get("finish_reason") not in {"stop", "length"} or not isinstance(metadata.get("usage"), dict):
                    raise APIError("Upstream lacks native chat metadata; deploy the matching Demo backend",
                                   status=502, code="upstream_contract_error")
                completed = (event, metadata)
                # Release the backend session before the gateway assigns another request.
                await ws.send(json.dumps({"type": "session.close", "reason": "turn_completed"}))
            elif kind == "session.closed" and completed is not None:
                await ws.wait_closed()
                return completed
            elif kind in {"error", "session.failed", "session.closed"}:
                error = event.get("error") or {}
                code = error.get("code") if isinstance(error, dict) else None
                raise APIError("Upstream chat failed" + (f": {code}" if code else ""),
                               status=503 if code == "queue_full" else 502, code="upstream_error")
    raise APIError("Upstream closed before completing the response", status=502, code="upstream_error")


def create_app(settings=None):
    settings = settings or Settings.from_env()
    parts = urlsplit(settings.upstream)
    if parts.scheme not in {"ws", "wss"} or not parts.netloc or parse_qs(parts.query).get("mode") != ["chat"]:
        raise ValueError("O5_CHAT_WS_URL must select the native mode=chat WebSocket endpoint")
    app = FastAPI(title="O5 native simplex · OpenAI benchmark API")

    @app.exception_handler(APIError)
    async def handle_error(request, exc):
        return JSONResponse(status_code=exc.status, content={"error": {
            "message": exc.message, "type": "invalid_request_error" if exc.status < 500 else "server_error",
            "param": exc.param, "code": exc.code}})

    def authorize(request):
        if settings.api_key and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + settings.api_key):
            raise APIError("Invalid API key", status=401, code="invalid_api_key")

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "native_hf_chat", "streaming": False,
                "model": settings.model, "upstream_checked": False}

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return {"object": "list", "data": [{"id": settings.model, "object": "model",
                "created": 0, "owned_by": "minicpm"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        authorize(request)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > settings.max_body_bytes:
                raise APIError("Request body is too large", status=413, code="request_too_large")
        try:
            body = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise APIError("Invalid JSON body") from exc
        payload = await asyncio.to_thread(translate, body, settings.model)
        task = asyncio.create_task(run_chat(payload, settings))
        started = time.monotonic()
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.2)
                if await request.is_disconnected():
                    raise APIError("Client disconnected", status=499, code="client_disconnected")
                if time.monotonic() - started >= settings.timeout_s:
                    raise APIError("Upstream chat timed out", status=504, code="upstream_timeout")
            event, metadata = task.result()
        except (ConnectionClosed, InvalidStatus, OSError, TimeoutError, ValueError) as exc:
            raise APIError("Cannot complete upstream chat", status=502, code="upstream_error") from exc
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        return {"id": "chatcmpl-o5-" + uuid.uuid4().hex, "object": "chat.completion",
                "created": int(time.time()), "model": settings.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": event.get("text", "")},
                             "finish_reason": metadata["finish_reason"]}],
                "usage": metadata["usage"], "o5": {"session_id": event.get("session_id"),
                "inference": "native_hf_chat", "elapsed_s": round(time.monotonic() - started, 3)}}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "18080")))
