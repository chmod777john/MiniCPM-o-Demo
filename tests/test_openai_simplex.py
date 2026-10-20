"""The bounded benchmark API contract; no model downloads or GPU required."""
import ast
import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import openai_app
from openai_app import APIError, Settings, create_app, translate
# Load the real parser without core/__init__ eagerly importing the GPU backend.
import dataclasses
import typing
import numpy as np
parser_ast = ast.parse(Path("py_backend/chat_util.py").read_text())
parser_nodes = [n for n in parser_ast.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in {"ChatRequestError", "WorkerChatRequest", "parse_worker_chat_request_message"}]
parser_ns = dict(dataclass=dataclasses.dataclass, field=dataclasses.field, np=np,
                 Dict=typing.Dict, Any=typing.Any, Optional=typing.Optional, base64=base64)
exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), "chat_util.py", "exec"), parser_ns)
parse_worker_chat_request_message = parser_ns["parse_worker_chat_request_message"]


def body(**extra):
    return {"model": "minicpm-o5", "messages": [{"role": "user", "content": "hello"}], **extra}


def test_benchmark_image_and_greedy():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    request = body(temperature=0.0, max_tokens=2048, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
        {"type": "text", "text": "What color?"}]}])
    payload = translate(request, "minicpm-o5")
    parsed = parse_worker_chat_request_message({"type": "chat.request", "payload": payload})
    assert parsed.max_new_tokens == 2048 and not parsed.do_sample
    assert parsed.generation_options == {"top_p": 0.8}
    assert parsed.return_completion_metadata and not parsed.generate_audio
    assert payload["messages"][0]["content"][0]["data"] == url.split(",")[1]
    assert not payload["use_tts_template"]


def test_preserve_history_and_parameters():
    history = [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "My name is Lin"},
               {"role": "assistant", "content": "OK"}, {"role": "user", "content": "My name?"}]
    payload = translate(body(messages=history, temperature=0.3, top_p=0.9, max_completion_tokens=77), "minicpm-o5")
    assert payload["messages"] == history
    parsed = parse_worker_chat_request_message({"type": "chat.request", "payload": payload})
    assert parsed.do_sample and parsed.generation_options == {"temperature": 0.3, "top_p": 0.9}
    assert parsed.max_new_tokens == 77


@pytest.mark.parametrize("extra", [
    {"stream": True}, {"stream": "false"}, {"n": 2}, {"n": True}, {"tools": [{"type": "function"}]},
    {"stop": ["END"]}, {"seed": 1}, {"frequency_penalty": 1}, {"temperature": -1},
    {"temperature": float("nan")}, {"top_p": 0}, {"max_tokens": 0}, {"max_tokens": 4097},
    {"max_tokens": True}, {"max_tokens": 12, "max_completion_tokens": 13},
    {"response_format": {"type": "json_object"}}, {"messages": []}, {"model": "other"},
    {"messages": [{"role": "developer", "content": "hi"}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.org/i.jpg"}}]}]},
])
def test_reject_unimplemented(extra):
    with pytest.raises(APIError):
        translate(body(**extra), "minicpm-o5")


def test_http_sdk_shape_and_auth(monkeypatch):
    async def generate(payload, settings):
        return {"text": "B", "session_id": "test-session"}, {"finish_reason": "length", "usage": {
            "prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101}}
    monkeypatch.setattr(openai_app, "run_chat", generate)
    client = TestClient(create_app(Settings(api_key="local-test")))
    assert client.get("/v1/models").status_code == 401
    headers = {"Authorization": "Bearer local-test"}
    assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "minicpm-o5"
    result = client.post("/v1/chat/completions", json=body(), headers=headers)
    from openai.types.chat import ChatCompletion
    parsed = ChatCompletion.model_validate(result.json())
    assert parsed.choices[0].message.content == "B"
    assert parsed.choices[0].finish_reason == "length"
    assert parsed.usage.total_tokens == 101
    assert client.post("/v1/chat/completions", content="{", headers=headers).status_code == 400
    assert client.post("/v1/chat/completions", json=body(stream=True), headers=headers).status_code == 400


def test_facade_observes_native_result_without_modification():
    # Execute only the real observer method so this CPU contract test needs no HF weights.
    source = ast.parse(Path("modeling/external_o5_facade.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "MiniCPMO")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    class Scalar:
        def __init__(self, value): self.value = value
        def item(self): return self.value
    class Sequence:
        def __init__(self, values): self.values = values
        def numel(self): return len(self.values)
        def __getitem__(self, idx): return Scalar(self.values[idx])
    class Base:
        def generate(self, **kwargs):
            self.seen = kwargs
            return self.native
    node = ast.ClassDef(name="Observer", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    ns = {"Base": Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "observer", "exec"), ns)
    model = ns["Observer"]()
    model._collect_chat_metadata = True
    model.terminators = ["EOS"]
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 99))
    for tokens, reason in (([7, 99], "stop"), ([7, 8, 9], "length")):
        outputs = SimpleNamespace(sequences=[Sequence(tokens)])
        model.native = (["original"], outputs)
        result = model.generate(input_ids=SimpleNamespace(shape=(1, 42)), max_new_tokens=3)
        assert result[0] is model.native[0] and result[1] is outputs
        assert model._last_chat_finish_reason == reason
        assert model._last_chat_token_stats["input_tokens"] == 42
        assert model._last_chat_token_stats["generated_tokens"] == len(tokens)
        assert model.seen["max_new_tokens"] == 3
