"""O45/O5 模型 primitive 的统一 Adapter。

共享 ``FcDuplexView`` 只依赖本模块的 Protocol。O45 与 O5 的 modeling、checkpoint
加载、TP2 和 LLM Graph 差异都必须停留在 Adapter 以下。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from minicpm_o5_sdk import O5SystemContent
from minicpm_o5_sdk.tokenizers.base import MiniCPMO5TokenizerAdapter

from core.fc_duplex.system_prefill import FcModelPrepareResult


@runtime_checkable
class FcDuplexModelAdapter(Protocol):
    """共享 View 所需的最小模型能力。"""

    model_family: Literal["o45", "o5"]
    tokenizer_target: Literal["o45_fc", "o5"]
    supports_stateless_resume: bool
    drops_unclassified_non_spoken_tokens: bool

    @property
    def protocol_tokenizer(self) -> MiniCPMO5TokenizerAdapter:
        """返回当前模型 target 对应的 SDK 0.0.5 tokenizer。"""

    @property
    def model_name(self) -> str:
        """返回只用于审计和 Resume identity 的模型名称。"""

    def prepare(
        self,
        *,
        system_content: O5SystemContent,
        tts_prompt_audio_path: str | None,
        generate_audio: bool | None,
    ) -> FcModelPrepareResult:
        """按 v3 canonical contract 初始化一个 FC Duplex Session。"""

    def streaming_prefill(self, **kwargs: Any) -> dict[str, Any]:
        """把当前 Unit 输入和工具结果写入模型。"""

    def streaming_spoken_generate(self, **kwargs: Any) -> dict[str, Any]:
        """生成当前 Unit 的 spoken slot。"""

    def streaming_non_spoken_generate(self, **kwargs: Any) -> dict[str, Any]:
        """推进一个或多个 non-spoken 模型 step。"""

    def finalize_unit(self) -> dict[str, Any]:
        """提交当前 Unit 并推进模型状态。"""

    def resume_boundary_status(self) -> dict[str, Any]:
        """返回当前边界是否允许 Stateless Resume。"""

    def replay_completed_unit(self, **kwargs: Any) -> dict[str, Any]:
        """确定性重放一个已提交 Unit。"""

    def decode_output_ids(self, **kwargs: Any) -> dict[str, Any]:
        """把内部 token 序列解析为结构化 FC 输出。"""

    def trace_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        """读取当前模型内部 trace。"""

    def dump_trace(self, **kwargs: Any) -> dict[str, Any]:
        """把当前模型内部 trace 落到指定路径。"""

    def cleanup(self) -> None:
        """释放当前 FC Session 状态。"""


class _BasePassthroughFcDuplexModelAdapter:
    """把既有 unified model 的 ``fc_duplex_*`` 方法适配为共享接口。"""

    model_family: Literal["o45", "o5"]
    tokenizer_target: Literal["o45_fc", "o5"]
    supports_stateless_resume: bool

    def __init__(self, model: Any) -> None:
        """保存一个已完成加载的 unified model。

        参数:
            model: 具备 ``fc_duplex_*`` primitive 的模型实例。
        """

        self._model = model

    @property
    def protocol_tokenizer(self) -> MiniCPMO5TokenizerAdapter:
        """返回模型 Capability 已绑定的 SDK tokenizer。"""

        capability = getattr(self._model, "fc_duplex", None)
        if capability is None:
            raise RuntimeError("FC duplex capability is not initialized")
        tokenizer = (
            getattr(capability, "protocol_tokenizer", None)
            or getattr(capability, "_sdk_tokenizer", None)
        )
        if tokenizer is None:
            raise RuntimeError("FC duplex capability has no protocol_tokenizer")
        target = getattr(tokenizer, "target", None)
        if target != self.tokenizer_target:
            raise RuntimeError(
                "FC tokenizer target mismatch: "
                f"adapter={self.tokenizer_target}, tokenizer={target}"
            )
        return tokenizer

    @property
    def model_name(self) -> str:
        """返回模型 config 中声明的名称。"""

        config = getattr(self._model, "config", None)
        return str(
            getattr(config, "_name_or_path", None)
            or getattr(self._model, "name_or_path", None)
            or "unknown"
        )

    def prepare(
        self,
        *,
        system_content: O5SystemContent,
        tts_prompt_audio_path: str | None,
        generate_audio: bool | None,
    ) -> FcModelPrepareResult:
        """调用模型 FC v3 prepare primitive。"""

        return dict(
            self._model.fc_duplex_prepare(
                system_content=system_content,
                tts_prompt_audio_path=tts_prompt_audio_path,
                generate_audio=generate_audio,
            )
        )

    def streaming_prefill(self, **kwargs: Any) -> dict[str, Any]:
        """调用模型 FC prefill primitive。"""

        return dict(self._model.fc_duplex_streaming_prefill(**kwargs))

    def streaming_spoken_generate(self, **kwargs: Any) -> dict[str, Any]:
        """调用模型 FC spoken primitive。"""

        return dict(self._model.fc_duplex_streaming_spoken_generate(**kwargs))

    def streaming_non_spoken_generate(self, **kwargs: Any) -> dict[str, Any]:
        """调用模型 FC non-spoken primitive。"""

        return dict(self._model.fc_duplex_streaming_non_spoken_generate(**kwargs))

    def finalize_unit(self) -> dict[str, Any]:
        """调用模型 FC Unit finalize primitive。"""

        return dict(self._model.fc_duplex_finalize_unit())

    def resume_boundary_status(self) -> dict[str, Any]:
        """返回 Capability 声明的 Resume 状态。"""

        if not self.supports_stateless_resume:
            return {
                "status": "unavailable",
                "reason": "resume_not_supported",
            }
        capability = getattr(self._model, "fc_duplex", None)
        method = getattr(capability, "resume_boundary_status", None)
        if method is None:
            return {"status": "available"}
        return dict(method())

    def replay_completed_unit(self, **kwargs: Any) -> dict[str, Any]:
        """重放一个 Unit；未承诺 Resume 的 Adapter 必须显式失败。"""

        if not self.supports_stateless_resume:
            raise NotImplementedError(
                f"{self.model_family} FC Adapter 尚未实现 Stateless Resume replay"
            )
        return dict(self._model.fc_duplex_replay_completed_unit(**kwargs))

    def decode_output_ids(self, **kwargs: Any) -> dict[str, Any]:
        """调用模型 FC token parser。"""

        return dict(self._model.fc_duplex_decode_output_ids(**kwargs))

    def trace_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        """返回模型 trace；模型不支持时返回明确状态。"""

        method = getattr(self._model, "fc_duplex_trace_snapshot", None)
        if method is None:
            return {"trace_supported": False, **kwargs}
        return dict(method(**kwargs))

    def dump_trace(self, **kwargs: Any) -> dict[str, Any]:
        """落盘模型 trace；模型不支持时显式失败。"""

        method = getattr(self._model, "fc_duplex_dump_trace", None)
        if method is None:
            raise NotImplementedError(
                f"{self.model_family} FC Adapter 不支持 dump_trace"
            )
        return dict(method(**kwargs))

    def cleanup(self) -> None:
        """释放模型 FC Session。"""

        self._model.fc_duplex_cleanup()


class O45FcDuplexModelAdapter(_BasePassthroughFcDuplexModelAdapter):
    """O45_FC target 的单卡模型 Adapter。"""

    model_family: Literal["o45"] = "o45"
    tokenizer_target: Literal["o45_fc"] = "o45_fc"
    supports_stateless_resume = True
    drops_unclassified_non_spoken_tokens = False


class O5FcDuplexModelAdapter(_BasePassthroughFcDuplexModelAdapter):
    """O5 target 的 MoE/TP2/LLM Graph 模型 Adapter。"""

    model_family: Literal["o5"] = "o5"
    tokenizer_target: Literal["o5"] = "o5"
    supports_stateless_resume = False
    # 忠实基线必须暴露模型协议违规，不能为了 Session 看起来可用而吞掉原始输出。
    drops_unclassified_non_spoken_tokens = False


def create_fc_duplex_model_adapter(
    *,
    model: Any,
    model_family: Literal["o45", "o5"],
) -> FcDuplexModelAdapter:
    """按部署 Profile 显式创建一个模型 Adapter。

    参数:
        model: 已完成 checkpoint 加载的 unified model。
        model_family: 部署 Profile 明确声明的 ``o45`` 或 ``o5``。

    返回:
        与模型族匹配的强类型 Adapter。
    """

    if model_family == "o45":
        return O45FcDuplexModelAdapter(model)
    if model_family == "o5":
        return O5FcDuplexModelAdapter(model)
    raise ValueError(f"unsupported FC model_family: {model_family}")
