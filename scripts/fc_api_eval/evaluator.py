"""TrainingData→Semantic API v2→token round-trip 的主 facade。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from minicpm_o5_sdk import O5DuplexTrainingData

from .models import (
    FcApiCheckpointProfile,
    FcApiEvaluationError,
    FcApiEvaluationErrorCategory,
    FcApiSemanticSessionResult,
    FcApiTrainingDataEvaluationResult,
    FcApiTrainingDataScenario,
)
from .semantic_v2_client import FcApiSemanticV2ClientError
from .token_reconstruction import FcApiTokenReconstructor
from .training_data_scenario import (
    REQUIRED_SDK_VERSION,
    build_training_data_scenario,
)


def set_session_generate_audio(
    session_init: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    """设置嵌套 ``session.init.payload.generate_audio``。

    参数:
        session_init: 完整上行 ``session.init`` 事件。
        enabled: 是否要求服务端生成 TTS waveform。
    """

    session_payload = session_init.get("payload")
    if not isinstance(session_payload, dict):
        raise ValueError("session.init 缺少 object payload")
    session_payload["generate_audio"] = enabled


class FcApiSemanticClientProtocol(Protocol):
    """Evaluator 依赖的最小 Semantic API client 接口。"""

    async def evaluate(
        self,
        scenario: FcApiTrainingDataScenario,
    ) -> FcApiSemanticSessionResult:
        """执行场景并返回完整会话。"""


class FcApiTrainingDataEvaluator:
    """MiniCPM O5 Demo 的无 GPU TrainingData API round-trip evaluator。"""

    def __init__(
        self,
        *,
        client: FcApiSemanticClientProtocol,
        profile: FcApiCheckpointProfile,
        data_root: Path,
        external_demo_root: Path = Path(
            "/user/sunweiyue/lib/"
            "minicpm-o-4_5-pytorch-simple-demo-merge-add-api"
        ),
        generate_audio: bool = False,
    ) -> None:
        """初始化 evaluator。

        参数:
            client: Semantic API v2 client；CPU 测试可注入 mock。
            profile: endpoint 的 checkpoint/Profile 元信息。
            data_root: TrainingData 媒体相对路径根目录。
            external_demo_root: 提供官方 resume canonicalizer 的外部 Demo 根目录。
            generate_audio: 是否要求服务端生成 spoken TTS waveform。
        """

        self.client = client
        self.profile = profile
        self.data_root = data_root
        self.generate_audio = generate_audio
        self.reconstructor = FcApiTokenReconstructor(
            external_demo_root=external_demo_root
        )

    async def evaluate(
        self,
        training_data: O5DuplexTrainingData,
    ) -> FcApiTrainingDataEvaluationResult:
        """执行 TrainingData→API→token round-trip 评测。

        参数:
            training_data: SDK 0.0.5 ``O5DuplexTrainingData``。

        返回:
            包含 SDK 版本、Profile、完整 history、逐轨/full exact、首差异、
            parser validate/round-trip 和错误分类的 Pydantic 结果。
        """

        if self.profile.sdk_version != REQUIRED_SDK_VERSION:
            return self._failure(
                data_id=training_data.data_id,
                category=(
                    FcApiEvaluationErrorCategory.SDK_VERSION_MISMATCH
                ),
                message=(
                    f"Profile SDK={self.profile.sdk_version}, "
                    f"required={REQUIRED_SDK_VERSION}"
                ),
            )
        try:
            scenario = build_training_data_scenario(
                training_data=training_data,
                profile=self.profile,
                data_root=self.data_root,
            )
            set_session_generate_audio(
                scenario.session_init,
                enabled=self.generate_audio,
            )
        except Exception as exc:
            category = _classify_scenario_error(exc)
            return self._failure(
                data_id=training_data.data_id,
                category=category,
                message=f"{type(exc).__name__}: {exc}",
            )

        try:
            session = await self.client.evaluate(scenario)
        except FcApiSemanticV2ClientError as exc:
            return FcApiTrainingDataEvaluationResult(
                data_id=training_data.data_id,
                sdk_version=scenario.sdk_version,
                profile=self.profile,
                errors=[
                    FcApiEvaluationError(
                        category=exc.category,
                        message=exc.message,
                        unit_index=exc.unit_index,
                        event_type=exc.event_type,
                    )
                ],
            )
        except Exception as exc:
            return self._failure(
                data_id=training_data.data_id,
                category=FcApiEvaluationErrorCategory.WEBSOCKET_ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )

        reconstruction = self.reconstructor.reconstruct(
            scenario=scenario,
            session=session,
            profile=self.profile,
        )
        return FcApiTrainingDataEvaluationResult(
            data_id=training_data.data_id,
            sdk_version=scenario.sdk_version,
            profile=self.profile,
            history=session.history,
            reconstruction=reconstruction,
        )

    def _failure(
        self,
        *,
        data_id: str | None,
        category: FcApiEvaluationErrorCategory,
        message: str,
    ) -> FcApiTrainingDataEvaluationResult:
        """构造 facade 的结构化失败结果。"""

        return FcApiTrainingDataEvaluationResult(
            data_id=data_id,
            sdk_version=self.profile.sdk_version,
            profile=self.profile,
            errors=[
                FcApiEvaluationError(
                    category=category,
                    message=message,
                )
            ],
        )


def _classify_scenario_error(
    exc: Exception,
) -> FcApiEvaluationErrorCategory:
    """根据场景构造错误的稳定前缀分类。"""

    message = str(exc)
    if "sdk_version_mismatch" in message:
        return FcApiEvaluationErrorCategory.SDK_VERSION_MISMATCH
    if "profile_target_mismatch" in message:
        return FcApiEvaluationErrorCategory.PROFILE_TARGET_MISMATCH
    if "profile_unit_policy_mismatch" in message:
        return (
            FcApiEvaluationErrorCategory.PROFILE_UNIT_POLICY_MISMATCH
        )
    if "unsupported_input_event" in message:
        return FcApiEvaluationErrorCategory.UNSUPPORTED_INPUT_EVENT
    if "unsupported_input_modality" in message:
        return FcApiEvaluationErrorCategory.UNSUPPORTED_INPUT_MODALITY
    if "reconstruction_unsupported" in message:
        return FcApiEvaluationErrorCategory.RECONSTRUCTION_UNSUPPORTED
    return FcApiEvaluationErrorCategory.REQUEST_BUILD_FAILED
