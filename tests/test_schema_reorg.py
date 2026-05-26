"""Compatibility checks for schema module organization."""

from pydantic import ValidationError
import pytest


def test_common_reexports_keep_legacy_imports():
    from core.schemas.common import GenerationConfig as LegacyGenerationConfig
    from core.schemas.common import Message as LegacyMessage
    from core.schemas.content import Message
    from core.schemas.options import GenerationConfig
    from core.schemas import GenerationConfig as PublicGenerationConfig
    from core.schemas import Message as PublicMessage

    assert LegacyGenerationConfig is GenerationConfig
    assert PublicGenerationConfig is GenerationConfig
    assert LegacyMessage is Message
    assert PublicMessage is Message


def test_existing_request_schemas_still_validate():
    from core.schemas import ChatRequest, DuplexConfig, Message, Role, StreamingRequest

    chat = ChatRequest(messages=[Message(role=Role.USER, content="你好")])
    assert chat.generation.temperature == 0.7

    streaming = StreamingRequest(
        session_id="s1",
        messages=[Message(role=Role.USER, content="继续")],
    )
    assert streaming.streaming.tts_sampling.top_k == 25

    duplex = DuplexConfig(force_listen_count=3, temperature=0.5)
    assert duplex.force_listen_count == 3
    assert duplex.temperature == 0.5


def test_backend_init_lifecycle_union():
    from pydantic import TypeAdapter

    from core.schemas.backend import BackendInitParams, FullDuplexBackendInit

    parsed = TypeAdapter(BackendInitParams).validate_python(
        {
            "mode": "full_duplex",
            "duplex": {"listen_prob_scale": 1.2, "force_listen_count": 2},
        }
    )

    assert isinstance(parsed, FullDuplexBackendInit)
    assert parsed.duplex.listen_prob_scale == 1.2


def test_backend_unary_reuses_existing_leaf_options():
    from core.schemas.backend import TurnBasedUnaryRequest, TurnBasedUnaryUsage

    req = TurnBasedUnaryRequest(
        messages=[{"role": "user", "content": "hello"}],
        generation={"temperature": 0.4, "top_k": 10},
        tts={"enabled": True, "sampling": {"top_p": 0.9}},
    )

    assert req.generation.temperature == 0.4
    assert req.tts.enabled is True
    assert req.tts.sampling.top_p == 0.9

    usage = TurnBasedUnaryUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    assert usage.total_tokens == 3


def test_backend_message_payload_is_typed_by_message_type():
    from core.schemas.backend import (
        BackendControl,
        BackendCloseMessage,
        BackendControlMessage,
        BackendInputMessage,
        FullDuplexStreamInput,
        TurnBasedStreamInput,
    )

    turn_input_msg = BackendInputMessage(
        type="input",
        payload={
            "mode": "turn_based",
            "messages": [{"role": "user", "content": "hello"}],
            "generation": {"temperature": 0.3},
        },
    )
    assert isinstance(turn_input_msg.payload, TurnBasedStreamInput)
    assert turn_input_msg.payload.generation.temperature == 0.3

    duplex_input_msg = BackendInputMessage(
        type="input",
        payload={
            "mode": "full_duplex",
            "content": [{"type": "audio", "data": "base64", "sample_rate": 16000}],
            "hints": {"force_listen": True},
        },
    )
    assert isinstance(duplex_input_msg.payload, FullDuplexStreamInput)
    assert duplex_input_msg.payload.hints.force_listen is True

    control_msg = BackendControlMessage(
        type="control",
        payload={"type": "metrics.get", "payload": {"detail": True}},
    )
    assert isinstance(control_msg.payload, BackendControl)
    assert control_msg.payload.type == "metrics.get"

    close_msg = BackendCloseMessage(type="close", payload={"reason": "done"})
    assert close_msg.payload.reason == "done"


def test_backend_schema_rejects_unknown_fields():
    from core.schemas.backend import TurnBasedBackendInit

    with pytest.raises(ValidationError):
        TurnBasedBackendInit(mode="turn_based", unknown=True)

