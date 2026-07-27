"""复用外部 Demo resume canonicalizer 重建 Semantic API v2 模型 token。

本模块不实现 API event parser。它调用外部 Demo 的
``build_fc_duplex_resume_plan`` 得到逐 Unit spoken/non-spoken IDs，再仅替换 GT
provenance 明确标记为 trainable 的连续区间，形成“确定性请求 scaffold + API 实际输出”
的完整候选序列。
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minicpm_o5_sdk.protocols.duplex import (
    O5DuplexParseRequest,
    O5DuplexParser,
    assert_duplex_parse_roundtrip,
)
from minicpm_o5_sdk.protocols.duplex.parsing import (
    render_duplex_token_diff,
)

from .models import (
    FcApiCheckpointProfile,
    FcApiEvaluationError,
    FcApiEvaluationErrorCategory,
    FcApiFirstTokenDiff,
    FcApiParserAuditResult,
    FcApiSemanticSessionResult,
    FcApiTokenReconstructionResult,
    FcApiTrackExactResult,
    FcApiTrainingDataScenario,
    TokenTrack,
)


_EXTERNAL_MODULE_NAME = "_fc_api_eval_external_fc_duplex_resume"


class FcApiTokenReconstructor:
    """Semantic API history→token 重建器。"""

    def __init__(self, *, external_demo_root: Path) -> None:
        """初始化重建器。

        参数:
            external_demo_root: 包含 ``core/fc_duplex_resume.py`` 的外部 Demo 根目录。
        """

        self.external_demo_root = external_demo_root
        self._resume_builder = _load_external_resume_builder(
            external_demo_root
        )

    def reconstruct(
        self,
        *,
        scenario: FcApiTrainingDataScenario,
        session: FcApiSemanticSessionResult,
        profile: FcApiCheckpointProfile,
    ) -> FcApiTokenReconstructionResult:
        """重建逐 Unit decode IDs，并尽力构建完整协议序列。

        参数:
            scenario: GT token、provenance span 和请求确定性 scaffold。
            session: Semantic API v2 的完整会话记录。
            profile: 当前 endpoint 的模型/Profile 身份。

        返回:
            per-track exact、full exact、首差异和 SDK parser 审计。
        """

        try:
            plan = self._resume_builder(
                protocol_version="fc-duplex-semantic-v2",
                model=profile.model,
                tokenizer_target=profile.tokenizer_target,
                tokenizer_fingerprint=scenario.tokenizer_fingerprint,
                through_unit_index=len(scenario.units) - 1,
                history=session.protocol_history,
            )
        except Exception as exc:
            category = _classify_resume_error(exc)
            return FcApiTokenReconstructionResult(
                full_reconstruction_supported=False,
                full_reconstruction_unsupported_reason=str(exc),
                errors=[
                    FcApiEvaluationError(
                        category=category,
                        message=f"{type(exc).__name__}: {exc}",
                        unit_index=getattr(exc, "unit_index", None),
                        detail={
                            "external_error_code": getattr(
                                exc, "code", None
                            )
                        },
                    )
                ],
            )

        actual_by_key: dict[tuple[int, str], list[int]] = {}
        for unit in plan.units:
            actual_by_key[(unit.unit_index, "spoken")] = list(
                unit.spoken_token_ids
            )
            non_spoken_ids = list(
                unit.non_spoken_token_ids
            )
            actual_by_key[(unit.unit_index, "non_spoken")] = (
                non_spoken_ids
            )
        per_track = _compare_per_track(
            scenario=scenario,
            actual_by_key=actual_by_key,
        )
        unit_exact = {
            unit.unit_index: all(
                result.exact
                for result in per_track
                if result.unit_index == unit.unit_index
            )
            for unit in scenario.expected_decode_units
        }

        try:
            reconstructed_ids = _splice_actual_decode_tracks(
                scenario=scenario,
                actual_by_key=actual_by_key,
            )
        except ValueError as exc:
            return FcApiTokenReconstructionResult(
                per_track=per_track,
                unit_exact=unit_exact,
                full_reconstruction_supported=False,
                full_reconstruction_unsupported_reason=str(exc),
                errors=[
                    FcApiEvaluationError(
                        category=(
                            FcApiEvaluationErrorCategory.RECONSTRUCTION_UNSUPPORTED
                        ),
                        message=str(exc),
                    )
                ],
            )

        parser = O5DuplexParser(profile.tokenizer_target)
        first_diff = first_token_diff(
            scenario.gt_input_ids,
            reconstructed_ids,
        )
        rendered_diff = render_duplex_token_diff(
            expected_ids=scenario.gt_input_ids,
            actual_ids=reconstructed_ids,
            parser=parser,
        )
        parser_audit, parser_errors = _audit_parser(
            parser=parser,
            scenario=scenario,
            reconstructed_ids=reconstructed_ids,
            profile=profile,
        )
        return FcApiTokenReconstructionResult(
            per_track=per_track,
            unit_exact=unit_exact,
            reconstructed_input_ids=reconstructed_ids,
            full_reconstruction_supported=True,
            full_exact=first_diff is None,
            first_diff=first_diff,
            rendered_diff=rendered_diff,
            parser_audit=parser_audit,
            errors=parser_errors,
        )


def first_token_diff(
    expected: list[int],
    actual: list[int],
) -> FcApiFirstTokenDiff | None:
    """返回两个 token 序列的首个差异。

    参数:
        expected: 期望 token IDs。
        actual: 实际 token IDs。

    返回:
        完全相等时返回 ``None``，否则返回首差异及两侧长度。
    """

    for index, (expected_id, actual_id) in enumerate(
        zip(expected, actual, strict=False)
    ):
        if expected_id != actual_id:
            return FcApiFirstTokenDiff(
                index=index,
                expected_token_id=expected_id,
                actual_token_id=actual_id,
                expected_length=len(expected),
                actual_length=len(actual),
            )
    if len(expected) == len(actual):
        return None
    index = min(len(expected), len(actual))
    return FcApiFirstTokenDiff(
        index=index,
        expected_token_id=(
            expected[index] if index < len(expected) else None
        ),
        actual_token_id=actual[index] if index < len(actual) else None,
        expected_length=len(expected),
        actual_length=len(actual),
    )


def _compare_per_track(
    *,
    scenario: FcApiTrainingDataScenario,
    actual_by_key: dict[tuple[int, str], list[int]],
) -> list[FcApiTrackExactResult]:
    """比较所有 Unit 的 spoken/non-spoken decode IDs。"""

    results: list[FcApiTrackExactResult] = []
    for expected_unit in scenario.expected_decode_units:
        track_pairs: tuple[tuple[TokenTrack, list[int]], ...] = (
            ("spoken", expected_unit.spoken_token_ids),
            ("non_spoken", expected_unit.non_spoken_token_ids),
        )
        for track, expected_ids in track_pairs:
            actual_ids = actual_by_key.get(
                (expected_unit.unit_index, track),
                [],
            )
            diff = first_token_diff(expected_ids, actual_ids)
            results.append(
                FcApiTrackExactResult(
                    unit_index=expected_unit.unit_index,
                    track=track,
                    expected_token_ids=expected_ids,
                    actual_token_ids=actual_ids,
                    exact=diff is None,
                    first_diff=diff,
                )
            )
    return results


def _splice_actual_decode_tracks(
    *,
    scenario: FcApiTrainingDataScenario,
    actual_by_key: dict[tuple[int, str], list[int]],
) -> list[int]:
    """用 API 实际轨替换 provenance 指定区间，保留其余确定性 scaffold。"""

    spans = sorted(
        scenario.gt_decode_spans,
        key=lambda span: span.start_token_index,
    )
    cursor = 0
    reconstructed: list[int] = []
    seen_keys: set[tuple[int, str]] = set()
    for span in spans:
        key = (span.unit_index, span.track)
        if key in seen_keys:
            raise ValueError(
                "full reconstruction unsupported: "
                f"Unit/track 出现多个 GT span: {key}"
            )
        if span.start_token_index < cursor:
            raise ValueError(
                "full reconstruction unsupported: GT decode spans overlap"
            )
        expected_unit = scenario.expected_decode_units[span.unit_index]
        expected_ids = (
            expected_unit.spoken_token_ids
            if span.track == "spoken"
            else expected_unit.non_spoken_token_ids
        )
        scaffold_slice = scenario.gt_input_ids[
            span.start_token_index : span.end_token_index_exclusive
        ]
        if scaffold_slice != expected_ids:
            raise ValueError(
                "full reconstruction unsupported: provenance span "
                f"与 expected decode 不一致: {key}"
            )
        if key not in actual_by_key:
            raise ValueError(
                "full reconstruction unsupported: API plan 缺少轨道 "
                f"{key}"
            )
        reconstructed.extend(
            scenario.gt_input_ids[cursor : span.start_token_index]
        )
        reconstructed.extend(actual_by_key[key])
        cursor = span.end_token_index_exclusive
        seen_keys.add(key)
    reconstructed.extend(scenario.gt_input_ids[cursor:])
    return reconstructed


def _audit_parser(
    *,
    parser: O5DuplexParser,
    scenario: FcApiTrainingDataScenario,
    reconstructed_ids: list[int],
    profile: FcApiCheckpointProfile,
) -> tuple[FcApiParserAuditResult, list[FcApiEvaluationError]]:
    """运行 SDK parser validate 与 replay round-trip。"""

    request = O5DuplexParseRequest(
        sequence=reconstructed_ids,
        sequence_format="token_ids",
        unit_policy=scenario.unit_policy,
        system=scenario.system,
        data_id=scenario.data_id,
        tool_serializer_name=profile.tool_serializer_name,
    )
    errors: list[FcApiEvaluationError] = []
    try:
        parser.validate(request)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append(
            FcApiEvaluationError(
                category=(
                    FcApiEvaluationErrorCategory.PARSER_VALIDATE_FAILED
                ),
                message=message,
                detail=(
                    exc.to_dict()
                    if hasattr(exc, "to_dict")
                    else None
                ),
            )
        )
        return (
            FcApiParserAuditResult(
                validate_ok=False,
                roundtrip_ok=False,
                error=message,
            ),
            errors,
        )
    try:
        roundtrip = assert_duplex_parse_roundtrip(parser, request)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        errors.append(
            FcApiEvaluationError(
                category=(
                    FcApiEvaluationErrorCategory.PARSER_ROUNDTRIP_FAILED
                ),
                message=message,
                detail=(
                    exc.to_dict()
                    if hasattr(exc, "to_dict")
                    else None
                ),
            )
        )
        return (
            FcApiParserAuditResult(
                validate_ok=True,
                roundtrip_ok=False,
                error=message,
            ),
            errors,
        )
    return (
        FcApiParserAuditResult(
            validate_ok=True,
            roundtrip_ok=roundtrip.exact_match,
            roundtrip_token_count=roundtrip.replay_token_count,
        ),
        errors,
    )


def _classify_resume_error(
    exc: Exception,
) -> FcApiEvaluationErrorCategory:
    """把外部 Demo resume 错误映射到 evaluator 稳定分类。"""

    code = str(getattr(exc, "code", "") or "")
    message = str(exc)
    if "non_spoken.end" in message:
        return FcApiEvaluationErrorCategory.MISSING_NON_SPOKEN_END
    if code == "model_or_tokenizer_mismatch":
        return FcApiEvaluationErrorCategory.PROFILE_OR_TARGET_MISMATCH
    if code in {
        "non_resumable_text_boundary",
        "unsupported_open_span",
        "unsupported_spoken_turn_state",
        "unsupported_deferred_close",
    } or "pending" in message.lower():
        return FcApiEvaluationErrorCategory.PENDING_TEXT
    return FcApiEvaluationErrorCategory.INCOMPLETE_HISTORY


def _load_external_resume_builder(
    external_demo_root: Path,
) -> Callable[..., Any]:
    """按文件路径加载外部 Demo canonical resume builder。"""

    module_path = (
        external_demo_root / "core" / "fc_duplex_resume.py"
    ).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(
            f"外部 resume canonicalizer 不存在: {module_path}"
        )
    existing = sys.modules.get(_EXTERNAL_MODULE_NAME)
    if existing is not None:
        builder = getattr(existing, "build_fc_duplex_resume_plan", None)
        if callable(builder):
            return builder
    spec = importlib.util.spec_from_file_location(
        _EXTERNAL_MODULE_NAME,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载外部 resume module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXTERNAL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    builder = getattr(module, "build_fc_duplex_resume_plan", None)
    if not callable(builder):
        raise ImportError(
            "外部 module 缺少 build_fc_duplex_resume_plan"
        )
    return builder
