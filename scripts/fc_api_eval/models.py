"""TrainingData→Semantic API v2→token round-trip 评测器的数据模型。

本模块只定义跨场景构造、WebSocket 采集、token 重建和最终报告共享的强类型结构，
不包含网络调用或 SDK grammar 解析逻辑。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from minicpm_o5_sdk import O5SystemContent, O5UnitPolicy
from pydantic import BaseModel, ConfigDict, Field


JsonObject = dict[str, Any]
TokenTrack = Literal["spoken", "non_spoken"]


class FcApiEvaluationErrorCategory(str, Enum):
    """评测失败的稳定分类。"""

    SDK_VERSION_MISMATCH = "sdk_version_mismatch"
    PROFILE_TARGET_MISMATCH = "profile_target_mismatch"
    PROFILE_UNIT_POLICY_MISMATCH = "profile_unit_policy_mismatch"
    REQUEST_BUILD_FAILED = "request_build_failed"
    UNSUPPORTED_INPUT_EVENT = "unsupported_input_event"
    UNSUPPORTED_INPUT_MODALITY = "unsupported_input_modality"
    WEBSOCKET_ERROR = "websocket_error"
    API_PROTOCOL_ERROR = "api_protocol_error"
    TOOL_CALL_MISMATCH = "tool_call_mismatch"
    INCOMPLETE_HISTORY = "incomplete_history"
    MISSING_NON_SPOKEN_END = "missing_non_spoken_end"
    PENDING_TEXT = "pending_text"
    PROFILE_OR_TARGET_MISMATCH = "profile_or_target_mismatch"
    RECONSTRUCTION_UNSUPPORTED = "reconstruction_unsupported"
    PARSER_VALIDATE_FAILED = "parser_validate_failed"
    PARSER_ROUNDTRIP_FAILED = "parser_roundtrip_failed"


class FcApiEvaluationError(BaseModel):
    """一条机器可读评测错误。"""

    model_config = ConfigDict(extra="forbid")

    category: FcApiEvaluationErrorCategory
    message: str
    unit_index: int | None = None
    event_type: str | None = None
    detail: JsonObject | None = None


class FcApiCheckpointProfile(BaseModel):
    """调用 Semantic API 时使用的 checkpoint/Profile 元信息。

    checkpoint 路径只作为运行产物元信息和 CLI 参数保存；通用 evaluator 不读取模型
    文件，也不把任何 checkpoint 路径写成类默认值。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    model: str
    checkpoint_path: Path
    training_job: str
    sdk_version: str = "0.0.5"
    tokenizer_target: Literal["o45_fc", "o5"] = "o45_fc"
    tool_serializer_name: str = "minicpm4_xml"
    non_spoken_scheduling: Literal["quality", "latency"] = "quality"
    expected_unit_policy: O5UnitPolicy | None = None


class FcApiHistoryEntry(BaseModel):
    """一条完整的 WebSocket 上行或下行记录。"""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    direction: Literal["up", "down"]
    event: JsonObject


class FcApiUnitInput(BaseModel):
    """一个 Unit 的确定性音频输入。"""

    model_config = ConfigDict(extra="forbid")

    unit_index: int = Field(ge=0)
    input_id: str
    audio_base64: str
    sample_rate: int = Field(gt=0)

    def to_frame(self) -> JsonObject:
        """返回可直接写入 WebSocket 的 ``input.append`` 帧。"""

        return {
            "type": "input.append",
            "input": {
                "input_id": self.input_id,
                "audio_base64": self.audio_base64,
                "sample_rate": self.sample_rate,
            },
        }


class FcApiExpectedDecodeUnit(BaseModel):
    """从 GT provenance 提取的单 Unit 两条模型 decode 轨。"""

    model_config = ConfigDict(extra="forbid")

    unit_index: int = Field(ge=0)
    spoken_token_ids: list[int] = Field(default_factory=list)
    non_spoken_token_ids: list[int] = Field(default_factory=list)


class FcApiDecodeSpan(BaseModel):
    """GT 完整序列中一段可由 API 实际 decode 轨替换的连续区间。"""

    model_config = ConfigDict(extra="forbid")

    unit_index: int = Field(ge=0)
    track: TokenTrack
    start_token_index: int = Field(ge=0)
    end_token_index_exclusive: int = Field(ge=0)


class FcApiExpectedToolCall(BaseModel):
    """按 TrainingData 离散 Gate 排序的工具调用。"""

    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=0)
    tool_call_id: str
    name: str
    arguments: JsonObject
    activation_unit_index: int = Field(ge=0)


class FcApiToolResponseSchedule(BaseModel):
    """一个必须在指定 GT Unit 前注入的原始字符串工具结果。"""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    send_before_unit_index: int = Field(ge=0)
    content: str


class FcApiTrainingDataScenario(BaseModel):
    """由一条 TrainingData 派生的完整无 GPU API 请求场景。"""

    model_config = ConfigDict(extra="forbid")

    data_id: str | None
    sdk_version: str
    tokenizer_target: Literal["o45_fc", "o5"]
    tokenizer_fingerprint: dict[str, str]
    unit_policy: O5UnitPolicy
    system: O5SystemContent | None
    session_init: JsonObject
    units: list[FcApiUnitInput]
    expected_decode_units: list[FcApiExpectedDecodeUnit]
    gt_decode_spans: list[FcApiDecodeSpan]
    expected_tool_calls: list[FcApiExpectedToolCall] = Field(default_factory=list)
    tool_responses: list[FcApiToolResponseSchedule] = Field(default_factory=list)
    gt_input_ids: list[int]


class FcApiSemanticSessionResult(BaseModel):
    """Semantic API v2 会话采集结果。"""

    model_config = ConfigDict(extra="forbid")

    history: list[FcApiHistoryEntry]
    protocol_history: list[JsonObject]
    session_created: JsonObject
    committed_unit_indices: list[int]
    api_to_gt_tool_call_id: dict[str, str] = Field(default_factory=dict)


class FcApiFirstTokenDiff(BaseModel):
    """首个 token 差异。"""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    expected_token_id: int | None
    actual_token_id: int | None
    expected_length: int = Field(ge=0)
    actual_length: int = Field(ge=0)


class FcApiTrackExactResult(BaseModel):
    """一个 Unit 的一条模型 decode 轨精确比较。"""

    model_config = ConfigDict(extra="forbid")

    unit_index: int = Field(ge=0)
    track: TokenTrack
    expected_token_ids: list[int]
    actual_token_ids: list[int]
    exact: bool
    first_diff: FcApiFirstTokenDiff | None = None


class FcApiParserAuditResult(BaseModel):
    """SDK parser validate 与 replay round-trip 审计结果。"""

    model_config = ConfigDict(extra="forbid")

    validate_ok: bool = False
    roundtrip_ok: bool = False
    roundtrip_token_count: int | None = None
    error: str | None = None


class FcApiTokenReconstructionResult(BaseModel):
    """Semantic API history 的 token 重建与 GT 比较结果。"""

    model_config = ConfigDict(extra="forbid")

    per_track: list[FcApiTrackExactResult] = Field(default_factory=list)
    unit_exact: dict[int, bool] = Field(default_factory=dict)
    reconstructed_input_ids: list[int] | None = None
    full_reconstruction_supported: bool
    full_reconstruction_unsupported_reason: str | None = None
    full_exact: bool | None = None
    first_diff: FcApiFirstTokenDiff | None = None
    rendered_diff: str | None = None
    parser_audit: FcApiParserAuditResult = Field(
        default_factory=FcApiParserAuditResult
    )
    errors: list[FcApiEvaluationError] = Field(default_factory=list)


class FcApiTrainingDataEvaluationResult(BaseModel):
    """主 facade 返回的完整评测报告。"""

    model_config = ConfigDict(extra="forbid")

    data_id: str | None
    sdk_version: str
    profile: FcApiCheckpointProfile
    history: list[FcApiHistoryEntry] = Field(default_factory=list)
    reconstruction: FcApiTokenReconstructionResult | None = None
    errors: list[FcApiEvaluationError] = Field(default_factory=list)

    @property
    def execution_ok(self) -> bool:
        """返回请求、会话和 token 重建是否没有结构性错误。

        SDK parser validate/round-trip 是独立诊断 gate。它们失败时仍可能已经从公共 API
        历史精确恢复完整 token，因此不覆盖请求执行与重建结论。
        """

        parser_categories = {
            FcApiEvaluationErrorCategory.PARSER_VALIDATE_FAILED,
            FcApiEvaluationErrorCategory.PARSER_ROUNDTRIP_FAILED,
        }
        blocking_reconstruction_errors = (
            []
            if self.reconstruction is None
            else [
                error
                for error in self.reconstruction.errors
                if error.category not in parser_categories
            ]
        )
        return not self.errors and (
            self.reconstruction is not None
            and not blocking_reconstruction_errors
        )
