"""FC Duplex runtime canonical generation batch 与 Unit checkpoint 测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.schemas.fc_duplex import (
    FcClosedSpan,
    FcGenerationProtocolOutput,
    FcGenerationStreamTerminationResult,
    FcGenerationTextDeltaOutput,
    FcGenerationTextPendingOutput,
    FcGenerationWarning,
    FcViewGenerationStep,
    NonSpokenStepGenerationFlag,
)
from minicpm_o5_sdk import (
    O5NoBudgetLimit,
    O5TokenizerID,
    O5UnitPolicy,
    load_builtin_tokenizer,
)
from py_backend.fc_duplex_runtime import FcDuplexSessionRuntime


class _FakeRuntimeBackend:
    """提供一个可完成单 Unit 的 backend stub。"""

    def __init__(self) -> None:
        """初始化可观测的 replay、stream sequence 与 prepare 状态。"""

        self.replayed_units: list[dict[str, Any]] = []
        self.next_stream_sequence: int | None = None
        self.prepare_fields: dict[str, Any] = {}

    def fc_duplex_prepare(self, **fields: Any) -> None:
        """记录 Runtime 透传给 backend 的初始化字段。"""

        self.prepare_fields = fields
        return

    def fc_duplex_prefill(self, **_: Any) -> None:
        return

    def fc_duplex_spoken_generate(self, **_: Any) -> Any:
        return SimpleNamespace(
            is_listen=True,
            is_speaking=False,
            spoken_text="",
            spoken_text_delta="",
            spoken_turn_eos=False,
            audio_waveform=None,
            audio_sample_rate=None,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=1,
                    stream_id="spoken_protocol",
                    track="spoken",
                    output=FcGenerationProtocolOutput(semantic_key="listen"),
                )
            ],
        )

    def fc_duplex_non_spoken_generate(self, **_: Any) -> Any:
        return SimpleNamespace(
            token_ids=[2],
            text="",
            text_delta="",
            close_reason="no_action",
            terminated=True,
            closed_spans=[],
            generation_flag=NonSpokenStepGenerationFlag.no_action,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=2,
                    stream_id="non_spoken_protocol",
                    track="non_spoken",
                    output=FcGenerationProtocolOutput(semantic_key="no_action"),
                )
            ],
        )

    def fc_duplex_finalize(self) -> None:
        return

    def fc_duplex_resume_boundary_status(self) -> dict[str, str]:
        return {"status": "available"}

    def fc_duplex_replay_completed_unit(self, **fields: Any) -> None:
        self.replayed_units.append(fields)

    def fc_duplex_restore_generation_stream_sequence(
        self,
        *,
        next_stream_sequence: int,
    ) -> None:
        self.next_stream_sequence = next_stream_sequence

    def fc_duplex_cleanup(self) -> None:
        return

    def fc_duplex_terminate_non_spoken_text_stream(
        self,
        *,
        reason: str,
    ) -> FcGenerationStreamTerminationResult:
        return FcGenerationStreamTerminationResult(
            generation_steps=[
                FcViewGenerationStep(
                    token_id=99,
                    stream_id="think_1",
                    track="non_spoken",
                    output=FcGenerationProtocolOutput(
                        semantic_key="non_spoken_budget_reached",
                        deferred_model_feed=True,
                    ),
                )
            ],
            warnings=[],
        )


class _ContinuingRuntimeBackend(_FakeRuntimeBackend):
    """按指定步数返回 continue，再自然产生 no_action 的 CPU backend。"""

    def __init__(self, *, terminate_after: int | None) -> None:
        """初始化可控 non-spoken backend。

        参数:
            terminate_after: 产生该数量 continue step 后返回 no_action；None 表示永不
                自然终止。
        """

        super().__init__()
        self.terminate_after = terminate_after
        self.non_spoken_calls = 0
        self.forced_budget_terminations = 0

    def fc_duplex_non_spoken_generate(self, **_: Any) -> Any:
        """返回一个可供 Runtime 循环控制测试使用的 non-spoken step。"""

        self.non_spoken_calls += 1
        if (
            self.terminate_after is not None
            and self.non_spoken_calls > self.terminate_after
        ):
            return SimpleNamespace(
                token_ids=[2],
                text="",
                text_delta="",
                close_reason="no_action",
                terminated=True,
                closed_spans=[],
                generation_flag=NonSpokenStepGenerationFlag.no_action,
                generation_steps=[],
                warnings=[],
            )
        return SimpleNamespace(
            token_ids=[3],
            text="",
            text_delta="",
            close_reason=None,
            terminated=False,
            closed_spans=[],
            generation_flag=(
                NonSpokenStepGenerationFlag.continue_non_spoken_generation
            ),
            generation_steps=[],
            warnings=[],
        )

    def fc_duplex_terminate_non_spoken_text_stream(
        self,
        *,
        reason: str,
    ) -> FcGenerationStreamTerminationResult:
        """记录有限 policy 触发的协议 budget 终止。"""

        self.forced_budget_terminations += 1
        return super().fc_duplex_terminate_non_spoken_text_stream(reason=reason)


@pytest.mark.asyncio
async def test_runtime_uses_explicit_profile_budgets_by_spoken_state() -> None:
    """Runtime 应按当前 Unit 的 spoken 决策选择 Profile 中对应 budget。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_budget",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime.prepare(
        {
            "checkpoint_profile_id": "profile_test",
            "config": {
                "non_spoken_scheduling": "quality",
                "non_spoken_budget_while_listening": 30,
                "non_spoken_budget_while_speaking": 15,
            },
        }
    )

    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_speaking=False),
        unit_index=0,
    ) == 30
    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_speaking=True),
        unit_index=0,
    ) == 15
    assert runtime.resume_identity["checkpoint_profile_id"] == "profile_test"
    assert runtime.resume_identity["non_spoken_budget_while_listening"] == 30
    assert runtime.resume_identity["non_spoken_budget_while_speaking"] == 15


@pytest.mark.asyncio
async def test_runtime_accepts_sdk_default_unit_policy_dump() -> None:
    """TrainingData 默认 UnitPolicy 的完整 JSON dump 应可直接初始化 Session。"""

    async def send(_: str, **__: Any) -> None:
        return

    policy = O5UnitPolicy(unit_sec=1.0)
    runtime = FcDuplexSessionRuntime(
        session_id="sess_sdk_default_policy",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    await runtime.prepare(
        {
            "unit_policy": policy.model_dump(mode="json"),
            "config": {"non_spoken_scheduling": "quality"},
        }
    )

    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_listen=True),
        unit_index=0,
    ) == 45
    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_speaking=True),
        unit_index=0,
    ) == 25
    assert runtime.resume_identity["unit_policy"] == policy.model_dump(mode="json")


@pytest.mark.asyncio
async def test_runtime_uses_complete_unit_policy_from_checkpoint_profile_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launcher 展开的完整 UnitPolicy 应成为 Session 的 Profile 事实。"""

    async def send(_: str, **__: Any) -> None:
        return

    policy = O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[2, 4],
        non_spoken_budgets_while_speaking=[1, 3],
    )
    monkeypatch.setenv("FC_DUPLEX_UNIT_POLICY_JSON", policy.model_dump_json())
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_unit_policy",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    await runtime.prepare(
        {"config": {"non_spoken_scheduling": "quality"}}
    )

    assert runtime.resume_identity["unit_policy"] == policy.model_dump(mode="json")
    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_listen=True),
        unit_index=1,
    ) == 4


@pytest.mark.asyncio
async def test_runtime_rejects_unit_policy_conflicting_with_profile_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session UnitPolicy 不得覆盖 launcher 绑定的完整 Profile 策略。"""

    async def send(_: str, **__: Any) -> None:
        return

    profile_policy = O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[30],
        non_spoken_budgets_while_speaking=[15],
    )
    session_policy = O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[31],
        non_spoken_budgets_while_speaking=[15],
    )
    monkeypatch.setenv(
        "FC_DUPLEX_UNIT_POLICY_JSON",
        profile_policy.model_dump_json(),
    )
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_unit_policy_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(
        RuntimeError,
        match="FC_DUPLEX_UNIT_POLICY_JSON",
    ):
        await runtime.prepare(
            {
                "unit_policy": session_policy.model_dump(mode="json"),
                "config": {"non_spoken_scheduling": "quality"},
            }
        )


@pytest.mark.asyncio
async def test_runtime_resolves_variable_policy_and_tail_repeat() -> None:
    """逐 Unit budget 应按 unit_index 读取，越界由 SDK 复用列表尾项。"""

    async def send(_: str, **__: Any) -> None:
        return

    policy = O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[2, 4],
        non_spoken_budgets_while_speaking=[1, 3],
    )
    runtime = FcDuplexSessionRuntime(
        session_id="sess_variable_policy",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime.prepare(
        {
            "unit_policy": policy.model_dump(mode="json"),
            "config": {"non_spoken_scheduling": "quality"},
        }
    )

    listen = SimpleNamespace(is_listen=True)
    speak = SimpleNamespace(is_speaking=True)
    assert runtime._select_non_spoken_budget(listen, unit_index=0) == 2
    assert runtime._select_non_spoken_budget(listen, unit_index=1) == 4
    assert runtime._select_non_spoken_budget(listen, unit_index=8) == 4
    assert runtime._select_non_spoken_budget(speak, unit_index=0) == 1
    assert runtime._select_non_spoken_budget(speak, unit_index=8) == 3


@pytest.mark.asyncio
async def test_runtime_tts_pad_uses_speaking_budget_without_is_speaking() -> None:
    """tts_pad 是 speaking 状态，即使 legacy is_speaking 字段为 False。"""

    async def send(_: str, **__: Any) -> None:
        return

    policy = O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[7],
        non_spoken_budgets_while_speaking=[3],
    )
    runtime = FcDuplexSessionRuntime(
        session_id="sess_tts_pad_policy",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime.prepare(
        {
            "unit_policy": policy.model_dump(mode="json"),
            "config": {"non_spoken_scheduling": "quality"},
        }
    )
    spoken = SimpleNamespace(
        is_speaking=False,
        generation_steps=[
            FcViewGenerationStep(
                token_id=1,
                stream_id="spoken_protocol",
                track="spoken",
                output=FcGenerationProtocolOutput(semantic_key="tts_pad"),
            )
        ],
    )

    assert runtime._select_non_spoken_budget(spoken, unit_index=0) == 3


@pytest.mark.asyncio
async def test_runtime_no_budget_limit_terminates_without_budget_marker() -> None:
    """O5NoBudgetLimit 应等待模型自然终止，不插入 budget_reached。"""

    events: list[dict[str, Any]] = []
    backend = _ContinuingRuntimeBackend(terminate_after=2)

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_no_budget",
        backend=backend,
        send=send,
    )
    runtime._non_spoken_scheduling = "quality"
    runtime._infrastructure_max_non_spoken_steps = 10

    await runtime._run_non_spoken_loop(
        input_id="u0",
        unit_index=0,
        pre_non_spoken_elapsed_ms=0.0,
        unit_budget=O5NoBudgetLimit(),
    )

    assert backend.non_spoken_calls == 3
    assert backend.forced_budget_terminations == 0
    assert events == [
        {
            "type": "response.non_spoken.end",
            "unit_index": 0,
            "reason": "no_action",
        }
    ]


@pytest.mark.asyncio
async def test_runtime_no_budget_limit_hits_infrastructure_safeguard() -> None:
    """无限样本 budget 达到基础设施上限时必须报错且不伪造协议 token。"""

    events: list[dict[str, Any]] = []
    backend = _ContinuingRuntimeBackend(terminate_after=None)

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_no_budget_safeguard",
        backend=backend,
        send=send,
    )
    runtime._non_spoken_scheduling = "quality"
    runtime._infrastructure_max_non_spoken_steps = 2

    with pytest.raises(RuntimeError, match="infrastructure max"):
        await runtime._run_non_spoken_loop(
            input_id="u0",
            unit_index=0,
            pre_non_spoken_elapsed_ms=0.0,
            unit_budget=O5NoBudgetLimit(),
        )

    assert backend.non_spoken_calls == 2
    assert backend.forced_budget_terminations == 0
    assert events == []


@pytest.mark.asyncio
async def test_runtime_rejects_missing_checkpoint_profile_budget() -> None:
    """未显式提供 Profile budget 时不能回退到某个 checkpoint 专属默认值。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_missing_profile_budget",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="non-spoken budgets"):
        await runtime.prepare({})


@pytest.mark.asyncio
async def test_runtime_uses_launcher_profile_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 未覆盖时应使用 launcher 注入的 Profile 身份和两类 budget。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("CHECKPOINT_PROFILE_ID", "profile_env")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_env",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    await runtime.prepare({})

    assert runtime.resume_identity["checkpoint_profile_id"] == "profile_env"
    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_speaking=False),
        unit_index=0,
    ) == 30
    assert runtime._select_non_spoken_budget(
        SimpleNamespace(is_speaking=True),
        unit_index=0,
    ) == 15


@pytest.mark.asyncio
async def test_runtime_rejects_session_budget_conflicting_with_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 参数不能静默覆盖 launcher 已绑定的 Checkpoint Profile。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="conflicts with Checkpoint Profile"):
        await runtime.prepare(
            {
                "config": {
                    "non_spoken_budget_while_listening": 12,
                    "non_spoken_budget_while_speaking": 12,
                }
            }
        )


@pytest.mark.asyncio
async def test_runtime_rejects_variable_policy_conflicting_with_scalar_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 UnitPolicy 不能绕过 launcher 已知的 legacy scalar checkpoint 事实。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    policy = O5UnitPolicy(
        unit_sec=1.0,
        non_spoken_budgets_while_listening=[30, 31],
        non_spoken_budgets_while_speaking=[15],
    )
    runtime = FcDuplexSessionRuntime(
        session_id="sess_policy_profile_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="conflicts with Checkpoint Profile"):
        await runtime.prepare(
            {"unit_policy": policy.model_dump(mode="json")}
        )


@pytest.mark.asyncio
async def test_runtime_rejects_checkpoint_profile_id_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 声明的 Profile ID 与进程绑定事实不一致时必须立即失败。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("CHECKPOINT_PROFILE_ID", "profile_env")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_id_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="conflicts with Checkpoint Profile"):
        await runtime.prepare(
            {
                "checkpoint_profile_id": "profile_request",
                "unit_policy": O5UnitPolicy(unit_sec=1.0).model_dump(
                    mode="json"
                ),
                "config": {"non_spoken_scheduling": "quality"},
            }
        )


@pytest.mark.asyncio
async def test_runtime_rejects_scheduling_conflicting_with_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 不能把 Profile 绑定的 quality 静默切成 latency。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_scheduling_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="conflicts with Checkpoint Profile"):
        await runtime.prepare({"config": {"non_spoken_scheduling": "latency"}})


@pytest.mark.asyncio
async def test_runtime_passes_fixed_tool_call_ids_to_backend() -> None:
    """评测固定 ID 应从 session.init evaluation 原样透传到 backend。"""

    async def send(_: str, **__: Any) -> None:
        return

    backend = _FakeRuntimeBackend()
    runtime = FcDuplexSessionRuntime(
        session_id="sess_fixed_ids",
        backend=backend,
        send=send,
    )
    await runtime.prepare(
        {
            "unit_policy": O5UnitPolicy(unit_sec=1.0).model_dump(mode="json"),
            "config": {"non_spoken_scheduling": "quality"},
            "evaluation": {
                "fixed_tool_call_ids": ["eval_call_a", "eval_call_b"]
            },
        }
    )

    assert backend.prepare_fields["fixed_tool_call_ids"] == [
        "eval_call_a",
        "eval_call_b",
    ]


@pytest.mark.asyncio
async def test_runtime_default_session_does_not_inject_fixed_tool_ids() -> None:
    """普通 Session 未提供 evaluation 时应保持 View 默认 ID 生成器。"""

    async def send(_: str, **__: Any) -> None:
        return

    backend = _FakeRuntimeBackend()
    runtime = FcDuplexSessionRuntime(
        session_id="sess_default_ids",
        backend=backend,
        send=send,
    )
    await runtime.prepare(
        {
            "unit_policy": O5UnitPolicy(unit_sec=1.0).model_dump(mode="json"),
            "config": {"non_spoken_scheduling": "quality"},
        }
    )

    assert backend.prepare_fields["fixed_tool_call_ids"] is None


@pytest.mark.asyncio
async def test_runtime_rejects_duplicate_fixed_tool_call_ids() -> None:
    """评测固定 ID 重复时必须在 Session 初始化阶段失败。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_duplicate_fixed_ids",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(ValueError, match="不能包含重复 ID"):
        await runtime.prepare(
            {
                "unit_policy": O5UnitPolicy(unit_sec=1.0).model_dump(
                    mode="json"
                ),
                "config": {"non_spoken_scheduling": "quality"},
                "evaluation": {
                    "fixed_tool_call_ids": ["eval_call_a", "eval_call_a"]
                },
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fixed_ids", [[], [""]])
async def test_runtime_rejects_empty_fixed_tool_call_ids(
    fixed_ids: list[str],
) -> None:
    """固定 ID 列表为空或包含空字符串时必须在初始化阶段失败。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_empty_fixed_ids",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(ValueError, match="fixed_tool_call_ids"):
        await runtime.prepare(
            {
                "unit_policy": O5UnitPolicy(unit_sec=1.0).model_dump(
                    mode="json"
                ),
                "config": {"non_spoken_scheduling": "quality"},
                "evaluation": {"fixed_tool_call_ids": fixed_ids},
            }
        )


@pytest.mark.asyncio
async def test_runtime_batches_safe_text_steps_without_exposing_token_ids() -> None:
    """Batch 应保留 pending/delta 次级边界，但公共 step 不包含 token_id。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_test",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    steps = [
        FcViewGenerationStep(
            token_id=100,
            stream_id="think_1",
            track="non_spoken",
            output=FcGenerationTextPendingOutput(),
        ),
        FcViewGenerationStep(
            token_id=101,
            stream_id="think_1",
            track="non_spoken",
            output=FcGenerationTextDeltaOutput(
                text="龘",
                source_step_count=2,
            ),
        ),
    ]

    await runtime._begin_block("think", unit_index=0)
    events.clear()
    await runtime._record_generation_steps(steps, unit_index=0, input_id="u0")
    await runtime._flush_generation_batch()

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "response.think.delta"
    assert event["unit_index"] == 0
    assert event["steps"] == [
        {"kind": "pending"},
        {"kind": "text", "text": "龘"},
    ]
    assert all("token_id" not in step for step in event["steps"])


@pytest.mark.asyncio
async def test_runtime_emits_available_unit_checkpoint_after_canonical_steps() -> None:
    """安全 Unit 应在 step batch 后发送带连续 event_index 的 checkpoint。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_test",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime.prepare(
        {
            "checkpoint_profile_id": "profile_test",
            "config": {
                "non_spoken_scheduling": "quality",
                "non_spoken_budget_while_listening": 30,
                "non_spoken_budget_while_speaking": 15,
            },
        }
    )
    await runtime._process_audio_payload(
        {
            "input_id": "u0",
            "audio_base64": "",
            "sample_rate": 16000,
        }
    )

    assert [event["type"] for event in events] == [
        "response.unit.started",
        "response.spoken.end",
        "response.non_spoken.end",
        "response.unit.committed",
    ]
    assert events[0]["tool_events"] == []
    assert events[-2] == {
        "type": "response.non_spoken.end",
        "unit_index": 0,
        "reason": "no_action",
    }
    checkpoint = events[-1]
    assert checkpoint["type"] == "response.unit.committed"
    assert checkpoint["unit_index"] == 0
    assert "non_spoken_end" not in checkpoint
    assert checkpoint["resume"] == {"status": "available"}


@pytest.mark.asyncio
async def test_runtime_resume_replays_public_history_and_continues_indices() -> None:
    """session.resume 应重建安全 Unit，并从 checkpoint 后的序号继续。"""

    events: list[dict[str, Any]] = []
    backend = _FakeRuntimeBackend()

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_resumed",
        backend=backend,
        send=send,
    )
    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "checkpoint_profile_id": "profile_test",
                "tokenizer_target": "o45_fc",
                "generate_audio": False,
                "config": {
                    "non_spoken_scheduling": "quality",
                    "non_spoken_budget_while_listening": 30,
                    "non_spoken_budget_while_speaking": 15,
                },
            },
        },
        {"type": "input.append", "input": {"input_id": "u0", "audio_base64": "AAAAAA=="}},
        {
            "type": "response.generation.step_batch",
            "event_index": 0,
            "batch_index": 0,
            "stream_id": "spoken_protocol",
            "track": "spoken",
            "steps": [
                {
                    "step_index": 0,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "listen"},
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 1,
            "batch_index": 1,
            "stream_id": "non_spoken_protocol",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 1,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "no_action"},
                }
            ],
        },
        {
            "type": "response.unit.committed",
            "event_index": 2,
            "unit_index": 0,
            "input_id": "u0",
            "last_step_index": 1,
            "resume": {"status": "available"},
        },
    ]

    await runtime.resume(
        {
            "protocol_version": "fc-duplex-resume-v1",
            "model": "minicpm-o-4.5",
            "tokenizer_target": "o45_fc",
            "tokenizer_fingerprint": {
                "vocab_hash": tokenizer.fingerprint.vocab_hash,
                "merges_hash": tokenizer.fingerprint.merges_hash,
            },
            "through_unit_index": 0,
            "history": history,
        }
    )

    assert len(backend.replayed_units) == 1
    replayed = backend.replayed_units[0]
    assert len(replayed["spoken_token_ids"]) == 1
    assert len(replayed["non_spoken_token_ids"]) == 1
    assert events[-1] == {
        "type": "session.resumed",
        "session_id": "sess_resumed",
        "through_unit_index": 0,
        "next_unit_index": 1,
    }
    assert runtime._generation_event_index == 3
    assert runtime._generation_step_index == 2
    assert runtime._generation_batch_index == 2
    assert backend.next_stream_sequence == 1


@pytest.mark.asyncio
async def test_runtime_rejects_missing_or_duplicate_input_id() -> None:
    """Live FC 输入必须有唯一 input_id，确保 checkpoint 能绑定真实处理 payload。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_input_ids",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    with pytest.raises(RuntimeError, match="requires input_id"):
        await runtime.enqueue_audio_input({"audio_base64": "AAAAAA=="})

    payload = {"input_id": "u0", "audio_base64": "AAAAAA=="}
    await runtime.enqueue_audio_input(payload)
    with pytest.raises(RuntimeError, match="duplicate"):
        await runtime.enqueue_audio_input(payload)
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_emits_budget_protocol_without_stream_end_warning() -> None:
    """Budget 只结束 Unit slot，不应产生 semantic stream end warning。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_warning",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    step = await runtime._build_deferred_budget_reached_step()
    await runtime._emit_step_events(step, input_id="u0", unit_index=0)

    assert runtime._unit_non_spoken_end == "budget_reached"
    assert {
        "type": "response.non_spoken.end",
        "unit_index": 0,
        "reason": "budget_reached",
    } in events
    assert not any(
        event["type"] in {
            "response.generation.step_batch",
            "response.output.sp_tokens",
        }
        for event in events
    )
    assert not any(event["type"] == "response.warning" for event in events)


@pytest.mark.asyncio
async def test_runtime_emits_non_spoken_end_exactly_once() -> None:
    """同一 Unit 的 eos 只能产生一个 non_spoken.end。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_non_spoken_end",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    eos_step = SimpleNamespace(
        token_ids=[],
        text_delta="",
        generation_steps=[],
        warnings=[],
        close_reason="eos",
        terminated=True,
        closed_spans=[],
    )

    await runtime._emit_step_events(eos_step, input_id="u0", unit_index=0)

    assert events == [
        {
            "type": "response.non_spoken.end",
            "unit_index": 0,
            "reason": "eos",
        }
    ]
    with pytest.raises(RuntimeError, match="duplicate response.non_spoken.end"):
        await runtime._emit_step_events(eos_step, input_id="u0", unit_index=0)


@pytest.mark.asyncio
async def test_runtime_rejects_unimplemented_non_spoken_end_reason() -> None:
    """Hold/Abort 尚未完整实现，不能进入当前 Public API。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_unsupported_non_spoken_end",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    unsupported = SimpleNamespace(
        token_ids=[],
        text_delta="",
        generation_steps=[],
        warnings=[],
        close_reason="hold",
        terminated=True,
        closed_spans=[],
    )

    with pytest.raises(RuntimeError, match="unsupported non-spoken end reason"):
        await runtime._emit_step_events(unsupported, input_id="u0", unit_index=0)


@pytest.mark.asyncio
async def test_runtime_does_not_leak_replacement_text_after_incomplete_bpe_warning() -> None:
    """Closed-span fallback 不得在 warning 后重新输出 lossy U+FFFD。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_lossy",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime._begin_block("think", unit_index=0)
    step = SimpleNamespace(
        token_ids=[],
        text_delta="",
        generation_steps=[],
        warnings=[
            FcGenerationWarning(
                code="incomplete_bpe_at_stream_end",
                stream_id="think_1",
                track="non_spoken",
                reason="think_end",
                message="文本边界包含未完成 BPE，公共 API 历史无法保证精确复现",
            )
        ],
        close_reason=None,
        terminated=False,
        closed_spans=[FcClosedSpan(type="think", text="\ufffd")],
    )

    await runtime._emit_step_events(step, input_id="u0", unit_index=0)

    assert any(event["type"] == "response.warning" for event in events)
    assert any(event["type"] == "response.think.end" for event in events)
    assert not any(
        event["type"] == "response.think.end"
        and "\ufffd" in str(event.get("full_text") or "")
        for event in events
    )


@pytest.mark.asyncio
async def test_detached_queue_runtime_error_invokes_session_fatal_callback() -> None:
    """后台 Unit 处理异常必须通知并关闭所属 Session，不能只留 task exception。"""

    fatal_errors: list[Exception] = []
    backend = _FakeRuntimeBackend()

    def fail_spoken(**_: Any) -> Any:
        raise RuntimeError("listen before spoken_turn_eos")

    backend.fc_duplex_spoken_generate = fail_spoken  # type: ignore[method-assign]

    async def send(_: str, **__: Any) -> None:
        return

    async def on_fatal(error: Exception) -> None:
        fatal_errors.append(error)

    runtime = FcDuplexSessionRuntime(
        session_id="sess_fatal",
        backend=backend,
        send=send,
        on_fatal=on_fatal,
    )
    await runtime.enqueue_audio_input(
        {
            "input_id": "u0",
            "audio_base64": "AAAAAA==",
            "sample_rate": 16000,
        }
    )
    assert runtime._queue_worker is not None
    await runtime._queue_worker

    assert len(fatal_errors) == 1
    assert "listen before spoken_turn_eos" in str(fatal_errors[0])
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_runtime_emits_explicit_processed_unit_tool_event_attribution() -> None:
    """Tool events 必须由 backend 显式绑定到实际处理 Unit，不能靠时序推断。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_tool_events",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    runtime._internal_to_api["fc_call_000001"] = "tc_000002"
    prefill = SimpleNamespace(
        tool_events=[
            {"type": "tool_started", "call_id": "fc_call_000001"},
            {
                "type": "tool_response",
                "call_id": "fc_call_000001",
                "content": "displayed",
            },
        ]
    )

    await runtime._emit_unit_input_events(
        prefill,
        unit_index=7,
        input_id="actual_processed_input",
    )

    assert events == [
        {
            "type": "response.unit.started",
            "input_id": "actual_processed_input",
            "unit_index": 7,
            "tool_events": [
                {
                    "type": "tool_started",
                    "tool_call_id": "tc_000002",
                },
                {
                    "type": "tool_result",
                    "tool_call_id": "tc_000002",
                },
            ],
        }
    ]
