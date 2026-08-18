from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.tracing import (
    DuplexTraceController,
    ForcingPolicy,
    MemoryTraceSink,
    ReplayReference,
    SessionBundleWriter,
    group_trace_events,
)
from py_backend.server import BackendProtocolSession, BackendServerState, _trace_groups
from core.processors.unified import DuplexView
from core.schemas.duplex import DuplexConfig
from tools.o5replay.compare import compare
from tools.o5replay.run_session import _is_tp2_worker
from tools.o5replay.session_io import RecordedSession, materialize_session_inputs, write_wav


class _FakeDecoder:
    def decode(self, logits, *args, **kwargs):
        return logits.argmax(dim=-1).long()

    def feed(self, embeds, return_logits=False):
        hidden = embeds.unsqueeze(0) if embeds.ndim == 2 else embeds
        logits = hidden.mean(dim=-1)
        return (logits, hidden) if return_logits else None


class _FakeFlowDecoder:
    def __init__(self):
        self.rand_noise = torch.arange(4, dtype=torch.float32)


class _FakeFlow:
    def __init__(self):
        self.decoder = _FakeFlowDecoder()


class _FakeAudioTokenizer:
    def __init__(self):
        self.flow = _FakeFlow()

    def stream(self, tokens, *args, **kwargs):
        return np.asarray(tokens, dtype="<i2").tobytes()


class _FakeTTS:
    def __init__(self, favored_token: int):
        self.audio_tokenizer = _FakeAudioTokenizer()
        self.favored_token = favored_token

    def generate_chunk(self, *args, **kwargs):
        steps = []
        for _ in range(2):
            probabilities = torch.full((2, 8), 0.01)
            probabilities[:, self.favored_token] = 0.93
            steps.append(torch.multinomial(probabilities, 1).reshape(-1))
        # The real TTS loop samples EOS once more but excludes it from the
        # returned content-token chunk.
        eos_probabilities = torch.full((2, 8), 0.01)
        eos_probabilities[:, 0] = 0.93
        torch.multinomial(eos_probabilities, 1)
        tokens = torch.stack(steps, dim=0).unsqueeze(0)
        return tokens, {"length": 2}


class _FakeModel:
    def __init__(self, favored_tts_token: int):
        self.tts = _FakeTTS(favored_tts_token)


class _FakeDuplex:
    def __init__(self, *, favored_llm_token: int, favored_tts_token: int, condition_bias: float):
        self.decoder = _FakeDecoder()
        self.model = _FakeModel(favored_tts_token)
        self.total_ids = []
        self.token2wav_buffer = [99]
        self.pre_lookahead = 1
        self.favored_llm_token = favored_llm_token
        self.condition_bias = condition_bias

    def _convert_results_to_tts_input(self, results):
        return torch.full((1, len(results) + 1, 4), self.condition_bias, dtype=torch.float32)

    def _generate_waveform_from_tokens(
        self,
        new_tokens,
        prompt_wav_path=None,
        is_last_chunk=False,
        force_flush=False,
    ):
        self.token2wav_buffer.extend(int(item) for item in new_tokens.reshape(-1).tolist())
        tokens = self.token2wav_buffer[:3]
        pcm = self.model.tts.audio_tokenizer.stream(tokens, last_chunk=is_last_chunk)
        self.token2wav_buffer = [] if is_last_chunk else self.token2wav_buffer[2:]
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32)

    def streaming_generate(self, *args, **kwargs):
        logits = torch.zeros(1, 16)
        logits[:, self.favored_llm_token] = 10
        selected = self.decoder.decode(logits)
        token_id = int(selected.item())
        self.total_ids.extend([token_id, 777])
        hidden = torch.full((1, 1, 4), float(token_id))
        condition = self._convert_results_to_tts_input([(token_id, hidden, False)])
        new_tokens, _cache = self.model.tts.generate_chunk(
            inputs_embeds=condition,
            text_start_pos=3,
            max_new_token=2,
            min_new_tokens=1,
        )
        waveform = self._generate_waveform_from_tokens(new_tokens, None, False, False)
        return {
            "text": f"token-{token_id}",
            "is_listen": False,
            "end_of_turn": False,
            "audio_waveform": waveform,
        }


def _run(controller: DuplexTraceController, duplex: _FakeDuplex, input_id: str = "unit-0"):
    controller.set_session("session-1")
    session_events = controller.drain()
    controller.set_unit(input_id, 0)
    result = duplex.streaming_generate()
    return result, session_events + controller.drain(input_id)


def test_duplex_config_restores_tensor_tts_temperature():
    model = torch.nn.Linear(1, 1)
    model.duplex = SimpleNamespace(tts_temperature=0.8)
    model._duplex_config = {}
    view = DuplexView(model, config=DuplexConfig(tts_temperature=0.6))

    view.apply_config_to_model()

    assert torch.is_tensor(model.duplex.tts_temperature)
    assert model.duplex.tts_temperature.shape == (1,)
    assert model.duplex.tts_temperature.device == next(model.parameters()).device
    assert model.duplex.tts_temperature.item() == pytest.approx(0.6)


def test_record_bundle_and_force_replay(tmp_path: Path):
    torch.manual_seed(0)
    reference_duplex = _FakeDuplex(favored_llm_token=5, favored_tts_token=3, condition_bias=1.5)
    reference_controller = DuplexTraceController(capture_mode="replay").install(reference_duplex)
    result, events = _run(reference_controller, reference_duplex)
    assert result["text"] == "token-5"
    assert {event["kind"] for event in events} >= {
        "llm.decode",
        "llm.accepted",
        "tts.condition",
        "tts.chunk",
        "tts.sample",
        "token2wav.call",
    }
    assert len([event for event in events if event["kind"] == "tts.sample"]) == 3
    reference_chunk = next(event for event in events if event["kind"] == "tts.chunk")
    assert reference_chunk["new_tokens"]["shape"] == [1, 2, 2]

    bundle = tmp_path / "recording"
    writer = SessionBundleWriter(bundle, source_implementation="canonical-test")
    serialized = writer.append(events)
    writer.close()
    assert (bundle / "model_trace.jsonl").is_file()
    assert list((bundle / "trace_tensors").glob("*.pt"))
    assert all(not torch.is_tensor(value) for event in serialized for value in _walk_values(event))

    reference_controller.uninstall()
    assert reference_duplex.decoder.decode.__func__ is _FakeDecoder.decode

    replay = ReplayReference.load(bundle)
    candidate_duplex = _FakeDuplex(favored_llm_token=9, favored_tts_token=6, condition_bias=8.0)
    candidate_duplex.model.tts.audio_tokenizer.flow.decoder.rand_noise.fill_(100)
    candidate_controller = DuplexTraceController(
        capture_mode="replay",
        reference=replay,
        forcing=ForcingPolicy.parse("all"),
    ).install(candidate_duplex)
    replay_result, replay_events = _run(candidate_controller, candidate_duplex)

    assert replay_result["text"] == "token-5"
    condition = next(event for event in replay_events if event["kind"] == "tts.condition")
    assert condition["teacher_forced"] is True
    assert condition["actual_condition"]["sha256"] != condition["used_condition"]["sha256"]
    chunk = next(event for event in replay_events if event["kind"] == "tts.chunk")
    assert chunk["new_tokens"]["tokens"] == reference_chunk["new_tokens"]["tokens"]
    replay_samples = [event for event in replay_events if event["kind"] == "tts.sample"]
    assert len(replay_samples) == 3
    assert all(event["teacher_forced"] for event in replay_samples)
    assert torch.equal(
        candidate_duplex.model.tts.audio_tokenizer.flow.decoder.rand_noise,
        torch.arange(4, dtype=torch.float32),
    )
    candidate_controller.uninstall()


def test_incomplete_bundle_manifest(tmp_path: Path):
    writer = SessionBundleWriter(tmp_path / "failed", source_implementation="test")
    writer.close(completed=False)

    manifest = json.loads((tmp_path / "failed" / "trace_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is False


def test_only_nonzero_tp2_rank_is_output_worker(monkeypatch):
    args = SimpleNamespace(target="demo-tp2")
    monkeypatch.setenv("RANK", "1")
    assert _is_tp2_worker(args) is True
    monkeypatch.setenv("RANK", "0")
    assert _is_tp2_worker(args) is False
    args.target = "demo-single"
    monkeypatch.setenv("RANK", "1")
    assert _is_tp2_worker(args) is False


def test_tts_sample_replay_rejects_chunk_step_drift():
    reference = ReplayReference([{
        "kind": "tts.sample",
        "input_id": "unit-0",
        "step": 0,
        "selected_token_ids": [3, 4],
    }])

    with pytest.raises(RuntimeError, match="sample step mismatch"):
        reference.next_tts_sample_tokens("unit-0", expected_step=1)


def test_reference_cursors_reset_for_a_new_session():
    reference = ReplayReference([{
        "kind": "llm.decode",
        "input_id": "unit-0",
        "selected_token_id": 7,
    }])
    assert reference.next_llm_token("unit-0") == 7
    with pytest.raises(RuntimeError, match="reference exhausted"):
        reference.next_llm_token("unit-0")

    reference.reset()
    assert reference.next_llm_token("unit-0") == 7


def test_grouping_and_t2w_ranges():
    duplex = _FakeDuplex(favored_llm_token=2, favored_tts_token=4, condition_bias=0.0)
    controller = DuplexTraceController(capture_mode="tokens").install(duplex)
    _result, events = _run(controller, duplex)
    grouped = group_trace_events(events, input_id="unit-0")
    assert grouped["llm"]
    assert grouped["tts"]
    assert grouped["token2wav"]

    call = next(event for event in events if event["kind"] == "token2wav.call")
    assert call["input_range"] == [0, 3]
    assert call["committed_range"] == [0, 2]
    assert call["lookahead_range"] == [2, 3]
    assert call["output_sample_range"] == [0, 3]

    assert _trace_groups(grouped, "llm", "tts") == {
        "schema": grouped["schema"],
        "input_id": "unit-0",
        "llm": grouped["llm"],
        "tts": grouped["tts"],
    }
    assert _trace_groups(grouped, "token2wav") == {
        "schema": grouped["schema"],
        "input_id": "unit-0",
        "token2wav": grouped["token2wav"],
    }
    controller.uninstall()


def test_recorded_session_materialization_and_comparison(tmp_path: Path):
    source = tmp_path / "source"
    blob_dir = source / "blob"
    blob_dir.mkdir(parents=True)
    audio = np.arange(16, dtype="<f4") / 16.0
    raw = audio.tobytes()
    (blob_dir / "0000.f32").write_bytes(raw)
    rows = [
        {
            "seq": 0,
            "dir": "up",
            "frame": {
                "type": "session.init",
                "payload": {"system_prompt": "test", "config": {"generate_audio": True}, "seed": 7},
            },
        },
        {
            "seq": 1,
            "dir": "up",
            "frame": {"type": "input.append", "input": {"input_id": "unit-a"}},
            "payload_trace": {
                "audio": {
                    "blob_f32": "@blob/0000.f32",
                    "sample_rate": 16000,
                },
                "force_listen": True,
                "max_slice_nums": 2,
            },
        },
    ]
    (source / "stream.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (source / "meta.json").write_text("{}", encoding="utf-8")
    session = RecordedSession(source)
    unit = session.units()[0]
    assert unit.input_id == "unit-a"
    assert unit.force_listen is True
    assert unit.max_slice_nums == 2
    assert np.array_equal(unit.audio, audio)

    copied = tmp_path / "copied"
    materialize_session_inputs(source, copied)
    assert (copied / "stream.jsonl").read_bytes() == (source / "stream.jsonl").read_bytes()
    assert (copied / "blob" / "0000.f32").read_bytes() == raw

    left = tmp_path / "left"
    right = tmp_path / "right"
    event = {
        "kind": "llm.decode",
        "input_id": "unit-a",
        "selected_token_id": 3,
        "local_argmax_token_id": 3,
        "logits": {"_tensor": torch.tensor([[1.0, 2.0]])},
    }
    for root in (left, right):
        writer = SessionBundleWriter(root, source_implementation="test")
        writer.append([event])
        writer.close()
        write_wav(root / "output_audio.wav", np.asarray([0.0, 0.5], dtype=np.float32))
    report = compare(left, right)
    assert report["tokens"]["llm.decode.selected_token_id"] == {"count": 1, "equal": 1}
    assert report["tensors"]["llm.decode.logits"]["bitwise_count"] == 1
    assert report["audio"]["combined"]["pcm16_bitwise"] is True


def test_gateway_recorder_keeps_inline_trace(tmp_path: Path):
    from gateway_modules.session_recording import SessionRecorder

    recorder = SessionRecorder("sess-test", "duplex", data_dir=str(tmp_path))
    trace = {"schema": "o5.session-trace.v1", "llm": [{"kind": "llm.decode", "selected_token_id": 7}]}
    frame = {"type": "response.output.delta", "kind": "text", "text": "ok", "trace": trace}
    externalized, payload_trace = recorder._externalize(frame)
    assert externalized["trace"] == trace
    assert payload_trace is None
    recorder.close("test")


def test_api_routes_unit_trace_to_text_and_audio_frames():
    class FakeWebSocket:
        def __init__(self):
            self.frames = []

        async def send_json(self, frame):
            self.frames.append(frame)

    class FakeBackend:
        spmd_is_driver = False

        def __init__(self):
            self.unit_ids = []
            self.finalized = False

        def set_trace_unit_id(self, input_id):
            self.unit_ids.append(input_id)

        def duplex_prefill(self, **_kwargs):
            return {"n_vision_images": 0}

        def duplex_generate(self, **_kwargs):
            return SimpleNamespace(
                is_listen=False,
                text="ok",
                audio_data=base64.b64encode(np.zeros(8, dtype=np.float32).tobytes()).decode("ascii"),
                end_of_turn=False,
                n_tokens=1,
                n_tts_tokens=2,
            )

        def drain_trace_events(self, input_id):
            if input_id is None:
                return None
            return {
                "schema": "o5.session-trace.v1",
                "input_id": input_id,
                "llm": [{"kind": "llm.decode", "selected_token_id": 7}],
                "tts": [{"kind": "tts.sample", "selected_token_ids": [1, 2]}],
                "token2wav": [{"kind": "token2wav.call", "input_token_ids": [1, 2]}],
            }

        def metrics(self):
            return {}

        def duplex_finalize(self):
            self.finalized = True

    async def exercise():
        backend = FakeBackend()
        websocket = FakeWebSocket()
        session = BackendProtocolSession(
            session_id="session-1",
            mode="full_duplex",
            backend=backend,
            ws=websocket,
            state=BackendServerState(backend),
        )
        session.initialized = True
        audio = base64.b64encode(np.zeros(16, dtype=np.float32).tobytes()).decode("ascii")
        await session._push_full_duplex({"input_id": "unit-1", "audio": audio})
        await session._drain_finalize()
        return backend, websocket.frames

    backend, frames = asyncio.run(exercise())
    text_frame = next(frame for frame in frames if frame.get("kind") == "text")
    audio_frame = next(frame for frame in frames if frame.get("kind") == "audio")
    assert set(text_frame["trace"]) >= {"llm", "tts"}
    assert "token2wav" not in text_frame["trace"]
    assert set(audio_frame["trace"]) >= {"token2wav"}
    assert "llm" not in audio_frame["trace"]
    assert backend.finalized is True
    assert backend.unit_ids == ["unit-1", None]


def _walk_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value
