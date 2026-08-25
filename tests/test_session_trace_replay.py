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
    debug_trace_events,
    DuplexTraceController,
    ForcingPolicy,
    MemoryTraceSink,
    ReplayReference,
    SessionBundleWriter,
)
from core.sampling import tts_argmax_scope
from py_backend.server import BackendProtocolSession, BackendServerState
from core.processors.pytorch_backend import PyTorchBackend
from core.processors.unified import DuplexView
from core.processors.unified import _place_model_preserving_float_buffers
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


class _TraceableTTS(_FakeTTS):
    def __init__(self):
        super().__init__(favored_token=3)
        self.head_code = torch.nn.ModuleList([
            torch.nn.Linear(4, 8, bias=False),
            torch.nn.Linear(4, 8, bias=False),
        ])

    def generate_chunk(self, *args, **kwargs):
        hidden = kwargs["inputs_embeds"]
        raw_logits = torch.stack([head(hidden)[:, -1] for head in self.head_code], dim=1)
        probabilities = torch.softmax(raw_logits.reshape(-1, raw_logits.shape[-1]), dim=-1)
        selected = torch.multinomial(probabilities, 1).reshape(1, 1, len(self.head_code))
        return selected, {"length": 1}


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


class _FakeFcCapability:
    """Minimal FC capability: a separate decoder and direct ``_sample`` call."""

    def __init__(self, *, favored_llm_token: int, decoder: _FakeDecoder):
        self.decoder = decoder
        self.favored_llm_token = favored_llm_token

    def _sample(self, logits, mode):
        del mode
        return int(torch.argmax(logits[0]).item())

    def _tts_condition(self, results):
        return torch.zeros((1, len(results) + 1, 4), dtype=torch.float32)

    def _spoken_audio(self, *args, **kwargs):
        del args, kwargs
        return {}

    def streaming_spoken_generate(self, max_tokens=24, decode_mode="greedy"):
        del max_tokens
        logits = torch.zeros((1, 16), dtype=torch.float32)
        logits[:, self.favored_llm_token] = 10
        token_id = self._sample(logits, decode_mode)
        self.decoder.feed(torch.full((1, 16), float(token_id)), return_logits=True)
        return {
            "spoken_ids": [token_id],
            "spoken_text": f"token-{token_id}",
            "text": f"token-{token_id}",
            "is_listen": False,
            "spoken_turn_eos": False,
        }

    def streaming_non_spoken_generate(self, max_tokens=1, decode_mode="greedy", close_reason=None):
        del max_tokens, decode_mode, close_reason
        return {"token_ids": [], "text": ""}


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


def test_model_placement_preserves_float_buffers():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    module.register_buffer("inv_freq", torch.ones(2, dtype=torch.float32))

    _place_model_preserving_float_buffers(module, "cpu")

    assert module.weight.dtype == torch.bfloat16
    assert module.inv_freq.dtype == torch.float32


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
    token2wav_call = next(event for event in events if event["kind"] == "token2wav.call")
    assert token2wav_call["output_pcm"]["shape"] == [3]
    assert torch.equal(token2wav_call["output_pcm"]["_tensor"], torch.tensor([99, 3, 3], dtype=torch.int16))
    assert token2wav_call["state_before"] == {"prompt": None, "flow": None, "hift": None}
    debug_events = debug_trace_events(events)
    debug_chunk = next(event for event in debug_events if event["kind"] == "tts.chunk")
    assert debug_chunk["token_ids"] == reference_chunk["new_tokens"]["tokens"]
    assert "new_tokens" not in debug_chunk

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


def test_tp2_driver_replay_does_not_add_rank_local_collective(monkeypatch):
    reference = ReplayReference([{
        "kind": "llm.decode",
        "input_id": "unit-0",
        "selected_token_id": 5,
    }])

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("driver-only replay must not issue a distributed collective")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", unexpected_collective)

    duplex = _FakeDuplex(favored_llm_token=9, favored_tts_token=6, condition_bias=8.0)
    controller = DuplexTraceController(
        capture_mode="replay",
        reference=reference,
        forcing=ForcingPolicy.parse("llm"),
        tp_driver=True,
    ).install(duplex)
    try:
        result, _events = _run(controller, duplex)
    finally:
        controller.uninstall()

    assert result["text"] == "token-5"


def test_fc_capability_replay_patches_direct_sample_and_decoder_feed():
    reference = ReplayReference([{
        "kind": "llm.decode",
        "input_id": "unit-0",
        "selected_token_id": 7,
    }])
    duplex = _FakeDuplex(favored_llm_token=5, favored_tts_token=3, condition_bias=1.5)
    fc = _FakeFcCapability(favored_llm_token=9, decoder=_FakeDecoder())
    controller = DuplexTraceController(
        capture_mode="replay",
        reference=reference,
        forcing=ForcingPolicy.parse("llm"),
    ).install(duplex, fc_duplex=fc)
    try:
        controller.set_session("session-1")
        controller.set_unit("unit-0", 0)
        result = fc.streaming_spoken_generate()
        events = controller.drain("unit-0")
    finally:
        controller.uninstall()

    assert result["spoken_ids"] == [7]
    assert [event["selected_token_id"] for event in events if event["kind"] == "llm.decode"] == [7]
    assert [event["token_ids"] for event in events if event["kind"] == "llm.accepted"] == [[7]]
    feed_events = [event for event in events if event["kind"] == "llm.feed"]
    assert len(feed_events) == 1
    assert feed_events[0]["hidden"]["shape"] == [1, 1, 16]


def test_session_manifest_keeps_deployment_settings(tmp_path: Path):
    bundle = tmp_path / "manifest"
    writer = SessionBundleWriter(
        bundle,
        source_implementation="demo-single",
        manifest_extra={
            "deployment": {
                "mode": "single_eager",
                "experts_implementation": "eager",
                "llm_graph": False,
                "llm_cache": 32768,
            }
        },
    )
    writer.close()

    manifest = json.loads((bundle / "trace_manifest.json").read_text())
    assert manifest["deployment"] == {
        "mode": "single_eager",
        "experts_implementation": "eager",
        "llm_graph": False,
        "llm_cache": 32768,
    }


def test_replay_mode_captures_tts_hidden_and_raw_logits():
    torch.manual_seed(0)
    duplex = _FakeDuplex(favored_llm_token=5, favored_tts_token=3, condition_bias=1.5)
    duplex.model.tts = _TraceableTTS()
    controller = DuplexTraceController(capture_mode="replay").install(duplex)

    _result, events = _run(controller, duplex)
    controller.uninstall()

    forwards = [event for event in events if event["kind"] == "tts.forward"]
    assert len(forwards) == 1
    assert forwards[0]["step"] == 0
    assert forwards[0]["phase"] == "prefill"
    assert forwards[0]["hidden"]["shape"] == [1, 1, 4]
    assert forwards[0]["logits"]["shape"] == [1, 2, 8]
    assert torch.is_tensor(forwards[0]["hidden"]["_tensor"])
    assert torch.is_tensor(forwards[0]["logits"]["_tensor"])


def test_tts_argmax_scope_preserves_trace_sampling(monkeypatch):
    monkeypatch.setenv("O5_TTS_ARGMAX", "1")
    duplex = _FakeDuplex(favored_llm_token=5, favored_tts_token=3, condition_bias=1.5)
    controller = DuplexTraceController(capture_mode="replay").install(duplex)
    try:
        with tts_argmax_scope(True):
            _result, events = _run(controller, duplex)
    finally:
        controller.uninstall()

    samples = [event for event in events if event["kind"] == "tts.sample"]
    assert [event["selected_token_ids"] for event in samples] == [[3, 3], [3, 3], [0, 0]]
    assert all(event["teacher_forced"] is False for event in samples)


def test_incomplete_bundle_manifest(tmp_path: Path):
    writer = SessionBundleWriter(tmp_path / "failed", source_implementation="test")
    writer.close(completed=False)

    manifest = json.loads((tmp_path / "failed" / "trace_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is False


def test_backend_tokens_mode_does_not_create_replay_sidecar(tmp_path: Path):
    class FakeController:
        def __init__(self):
            self.session_id = None
            self.events = [{"kind": "llm.chunk", "input_id": "unit-0", "token_ids": [7]}]

        def set_session(self, session_id):
            self.session_id = session_id

        def drain(self, input_id=None):
            events, self.events = self.events, []
            return events

    backend = PyTorchBackend(model_path="unused", gpu_id=0)
    backend._trace_controller = FakeController()
    backend._trace_capture_mode = "tokens"
    backend._token_trace_dir = tmp_path

    backend.set_trace_session_id("session-1")
    events = backend.drain_trace_events("unit-0")

    assert backend._trace_writer is None
    assert not (tmp_path / "session-1").exists()
    assert events == [{"kind": "llm.chunk", "input_id": "unit-0", "token_ids": [7]}]


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


def test_legacy_chunk_reference_without_input_ids_uses_global_order():
    reference = ReplayReference([
        {"kind": "llm.chunk", "sampled_token_ids": [5]},
        {
            "kind": "tts.chunk",
            "sampled_token_ids": [[3, 4], [0, 0]],
        },
        {"kind": "llm.chunk", "sampled_token_ids": [6]},
        {
            "kind": "tts.chunk",
            "sampled_token_ids": [[7, 8]],
        },
    ])

    assert reference.next_llm_token("unit-000000") == 5
    assert reference.next_llm_token("unit-000001") == 6
    assert reference.next_tts_sample_tokens("unit-000000", expected_step=0) == [3, 4]
    assert reference.next_tts_sample_tokens("unit-000000", expected_step=1) == [0, 0]
    assert reference.next_tts_sample_tokens("unit-000001", expected_step=0) == [7, 8]


def test_tokens_mode_emits_minimal_chunk_events_and_t2w_ranges():
    duplex = _FakeDuplex(favored_llm_token=2, favored_tts_token=4, condition_bias=0.0)
    controller = DuplexTraceController(capture_mode="tokens").install(duplex)
    _result, events = _run(controller, duplex)
    assert [event["kind"] for event in events] == ["tts.chunk", "t2w.chunk", "llm.chunk"]

    llm = next(event for event in events if event["kind"] == "llm.chunk")
    assert llm["token_ids"] == [2, 777]
    assert llm["sampled_token_ids"] == [2]
    assert llm["is_listen"] is False
    assert llm["end_of_turn"] is False

    tts = next(event for event in events if event["kind"] == "tts.chunk")
    assert tts["source_llm_token_ids"] == [2]
    assert len(tts["token_ids"]) == 4
    assert [token for step in tts["sampled_token_ids"][:-1] for token in step] == tts["token_ids"]
    assert tts["sampled_token_ids"][-1] == [0, 0]

    call = next(event for event in events if event["kind"] == "t2w.chunk")
    assert call["input_range"] == [0, 3]
    assert call["committed_range"] == [0, 2]
    assert call["lookahead_range"] == [2, 3]
    assert call["output_sample_range"] == [0, 3]
    assert [event["kind"] for event in debug_trace_events(events)] == ["llm.chunk", "tts.chunk", "t2w.chunk"]
    forbidden = {"shape", "dtype", "numel", "sha256", "_tensor", "probabilities"}
    assert not forbidden.intersection(_walk_keys(events))
    controller.uninstall()


def test_chunk_debug_reference_can_teacher_force_llm_and_tts(tmp_path: Path):
    torch.manual_seed(0)
    reference_duplex = _FakeDuplex(favored_llm_token=5, favored_tts_token=3, condition_bias=1.5)
    reference_controller = DuplexTraceController(capture_mode="tokens").install(reference_duplex)
    reference_result, reference_events = _run(reference_controller, reference_duplex)
    reference_controller.uninstall()

    debug_frames = debug_trace_events(reference_events)
    session = tmp_path / "token-session"
    session.mkdir()
    (session / "stream.jsonl").write_text(
        "".join(
            json.dumps({"dir": "down", "frame": {"type": "debug", **frame}}) + "\n"
            for frame in debug_frames
        ),
        encoding="utf-8",
    )

    replay = ReplayReference.load(session)
    candidate_duplex = _FakeDuplex(favored_llm_token=9, favored_tts_token=6, condition_bias=8.0)
    candidate_controller = DuplexTraceController(
        capture_mode="tokens",
        reference=replay,
        forcing=ForcingPolicy.parse("llm,tts-token"),
    ).install(candidate_duplex)
    replay_result, replay_events = _run(candidate_controller, candidate_duplex)
    candidate_controller.uninstall()

    assert replay_result["text"] == reference_result["text"] == "token-5"
    reference_tts = next(event for event in reference_events if event["kind"] == "tts.chunk")
    replay_tts = next(event for event in replay_events if event["kind"] == "tts.chunk")
    assert replay_tts["token_ids"] == reference_tts["token_ids"]
    assert replay_tts["sampled_token_ids"] == reference_tts["sampled_token_ids"]
    assert [event["kind"] for event in replay_events] == ["tts.chunk", "t2w.chunk", "llm.chunk"]
    forbidden = {"shape", "dtype", "numel", "sha256", "_tensor", "probabilities"}
    assert not forbidden.intersection(_walk_keys(replay_events))


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


def test_comparison_reads_chunk_debug_from_gateway_stream(tmp_path: Path):
    frames = [
        {"type": "debug", "kind": "llm.chunk", "input_id": "unit-a", "token_ids": [3], "is_listen": False, "end_of_turn": False},
        {"type": "debug", "kind": "tts.chunk", "input_id": "unit-a", "source_llm_token_ids": [3], "token_ids": [10, 11]},
        {
            "type": "debug",
            "kind": "t2w.chunk",
            "input_id": "unit-a",
            "input_token_ids": [10, 11],
            "input_range": [0, 2],
            "committed_range": [0, 1],
            "lookahead_range": [1, 2],
            "output_sample_range": [0, 8],
            "last_chunk": False,
        },
    ]
    for root in (tmp_path / "left-debug", tmp_path / "right-debug"):
        root.mkdir()
        (root / "stream.jsonl").write_text(
            "".join(json.dumps({"dir": "down", "frame": frame}) + "\n" for frame in frames),
            encoding="utf-8",
        )

    report = compare(tmp_path / "left-debug", tmp_path / "right-debug")

    assert report["event_counts"] == {"left": 3, "right": 3, "common": 3, "left_only": 0, "right_only": 0}
    assert report["tokens"]["llm.chunk.token_ids"] == {"count": 1, "equal": 1}
    assert report["tokens"]["tts.chunk.token_ids"] == {"count": 1, "equal": 1}
    assert report["tokens"]["t2w.chunk.input_token_ids"] == {"count": 1, "equal": 1}


def test_comparison_counts_distribution_argmax_reversals(tmp_path: Path):
    left = tmp_path / "left-reversal"
    right = tmp_path / "right-reversal"
    events = (
        {
            "kind": "tts.forward",
            "input_id": "unit-a",
            "hidden": {"_tensor": torch.tensor([[[1.0, 2.0]]])},
            "logits": {"_tensor": torch.tensor([[[0.0, 2.0, 1.0]]])},
        },
        {
            "kind": "tts.forward",
            "input_id": "unit-a",
            "hidden": {"_tensor": torch.tensor([[[1.0, 2.0]]])},
            "logits": {"_tensor": torch.tensor([[[3.0, 0.0, 1.0]]])},
        },
    )
    for root, event in zip((left, right), events):
        writer = SessionBundleWriter(root, source_implementation="test")
        writer.append([event])
        writer.close()

    report = compare(left, right)
    logits = report["tensors"]["tts.forward.logits"]
    assert logits["argmax_equal"] == 0
    assert logits["argmax_total"] == 1
    assert logits["argmax_reversals"] == 1


def test_gateway_recorder_keeps_debug_frame(tmp_path: Path):
    from gateway_modules.session_recording import SessionRecorder

    recorder = SessionRecorder("sess-test", "duplex", data_dir=str(tmp_path))
    frame = {
        "type": "debug",
        "kind": "llm.chunk",
        "input_id": "unit-1",
        "unit_index": 0,
        "turn_id": 0,
        "token_ids": [7],
    }
    externalized, payload_trace = recorder._externalize(frame)
    assert externalized == frame
    assert payload_trace is None
    recorder.close("test")


def test_api_sends_standalone_debug_frames():
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
            self.trace_drained = False

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
            if input_id is None or self.trace_drained:
                return None
            self.trace_drained = True
            identity = {"session_id": "session-1", "input_id": input_id, "unit_index": 0, "turn_id": 0}
            return [
                {**identity, "kind": "llm.chunk", "token_ids": [7], "is_listen": False, "end_of_turn": False},
                {**identity, "kind": "tts.chunk", "source_llm_token_ids": [7], "token_ids": [1, 2]},
                {
                    **identity,
                    "kind": "t2w.chunk",
                    "input_token_ids": [1, 2],
                    "committed_range": [0, 1],
                    "lookahead_range": [1, 2],
                    "output_sample_range": [0, 8],
                },
            ]

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
    business_frames = [frame for frame in frames if frame["type"] != "debug"]
    debug_frames = [frame for frame in frames if frame["type"] == "debug"]
    assert all("trace" not in frame for frame in business_frames)
    assert [frame["kind"] for frame in debug_frames] == ["llm.chunk", "tts.chunk", "t2w.chunk"]
    assert all(frame["input_id"] == "unit-1" for frame in debug_frames)
    assert all("server_send_ts" in frame for frame in debug_frames)
    assert backend.finalized is True
    assert backend.unit_ids == ["unit-1", None]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _walk_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value
