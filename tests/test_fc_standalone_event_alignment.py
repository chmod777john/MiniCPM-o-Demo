"""standalone input_event 的训推一致性：推理侧注入的 token 必须与 SDK tokenize() 逐个相同。

做法：同一份 TrainingData，SDK 按 Unit 产出 input_event 槽的 token；把该 Unit 的同一组事件
（tool created / tool response / standalone，按 SDK arrangement 的顺序）交给 O5 推理侧渲染器
``FcDuplexCapability._event_ids``，两边逐 token 比对。

- 自包含用例：在内置 case_3_01（一次 get_weather）里按 sub_agent 训练数据的真实写法插入
  standalone 事件——挂在工具调用结束之后、带秒级 offset——覆盖「工具结果之前 / 与工具结果
  同 Unit / 之后」三种位置，以及换行、引号、emoji 等普通文本。
- 真实数据用例：直接取 sub_agent 训练样本；本机没有该数据时跳过。

O45 已停止维护（仓库根目录 CLAUDE.md），standalone 只做 O5。
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import pytest

from minicpm_o5_sdk import O5DuplexTrainingData, O5TokenizerID
from minicpm_o5_sdk.cases.loader import load_case_structure
from minicpm_o5_sdk.protocols.duplex.training_data.events import (
    O5StandaloneEvent,
    O5ToolCreatedEvent,
    O5ToolResponseEvent,
)

from modeling.o5.modeling_minicpmo_unified import FcDuplexCapability

SUB_AGENT_BUNDLE = Path(
    "/user/sunweiyue/dataset/o5_sdk_training_data/sub_agent/"
    "agent_duplex_sub_agent_0929_min3k_max8k_zh_nothink_by_swy"
)


@pytest.fixture(scope="module")
def renderer() -> FcDuplexCapability:
    """只挂 SDK tokenizer / registry 的渲染器，不加载模型权重。"""

    cap = object.__new__(FcDuplexCapability)
    cap.tool_format = "minicpm4_xml"
    cap.ids = {}
    cap.id2name = {}
    cap._registry = None
    cap._ensure_protocol()
    return cap


def _standalone_segment(text: str, *, call_segment_index: int, offset_sec: float) -> dict:
    """sub_agent 训练数据里 standalone 的真实写法：挂在发起子任务的 tool_call 结束之后。"""

    return {
        "start_trigger": {
            "refs": [{
                "type": "text",
                "track": "ai_non_spoken",
                "segment_index": call_segment_index,
                "at": "end",
                "char_index": None,
                "offset_sec": offset_sec,
            }],
            "mode": "deferrable",
        },
        "event": {"kind": "standalone", "contents": [{"kind": "text", "text": text}]},
    }


def _sdk_input_event_tokens_by_unit(tokenized) -> dict[int, list[int]]:
    """SDK 产物里每个 Unit 的 input_event 槽 token（含槽首尾边界）。"""

    by_unit: dict[int, list[int]] = defaultdict(list)
    for item in tokenized.token_provenance:
        track = item["track"] if isinstance(item, dict) else item.track
        if track != "input_event":
            continue
        unit = item["unit_index"] if isinstance(item, dict) else item.unit_index
        token_id = item["token_id"] if isinstance(item, dict) else item.token_id
        by_unit[unit].append(int(token_id))
    return dict(by_unit)


def _runtime_events_by_unit(arrangement) -> dict[int, list[dict]]:
    """把 SDK arrangement 的事件翻成推理侧渲染器吃的 dict（与 view 层组装的形态一致）。"""

    by_unit: dict[int, list[dict]] = defaultdict(list)
    for segment in arrangement.tracks.input_event.segments:
        event = segment.event
        unit = segment.timeline.start_unit_index
        if isinstance(event, O5ToolCreatedEvent):
            by_unit[unit].append({"type": "tool_started", "call_id": event.tool_call_id})
        elif isinstance(event, O5ToolResponseEvent):
            by_unit[unit].append({
                "type": "tool_response",
                "call_id": event.tool_call_id,
                "content": event.contents[0].text,
            })
        elif isinstance(event, O5StandaloneEvent):
            by_unit[unit].append({"type": "standalone_event", "content": event.contents[0].text})
        else:  # pragma: no cover - SDK 新增事件类型时显式失败
            raise AssertionError(f"未覆盖的 input_event 类型: {type(event).__name__}")
    return dict(by_unit)


def _assert_aligned(renderer, training_data: O5DuplexTrainingData) -> tuple[int, int]:
    result = training_data.tokenize(
        tokenizer_id=O5TokenizerID.O5,
        prepend_empty_think_before_action=False,
    )
    sdk = _sdk_input_event_tokens_by_unit(result.tokenized_data)
    events = _runtime_events_by_unit(result.arrangement)
    assert sorted(sdk) == sorted(events), "SDK 与 arrangement 对「哪些 Unit 有 input_event」不一致"
    standalone_units = 0
    for unit, sdk_ids in sdk.items():
        ours = renderer._event_ids(events[unit])
        assert ours == sdk_ids, (
            f"Unit {unit} 的 input_event token 与 SDK 不一致\n"
            f"  events: {json.dumps(events[unit], ensure_ascii=False)[:300]}\n"
            f"  sdk : {sdk_ids}\n  ours: {ours}"
        )
        standalone_units += any(e["type"] == "standalone_event" for e in events[unit])
    return len(sdk), standalone_units


def test_standalone_tokens_match_sdk_on_augmented_builtin_case(renderer) -> None:
    """自包含：在 case_3_01 里插入 standalone，覆盖三种位置与特殊字符。"""

    structure = copy.deepcopy(load_case_structure("case_3_01"))
    segments = structure["tracks"]["input_event"]["segments"]
    response_offset = segments[0]["start_trigger"]["refs"][0]["offset_sec"]
    segments.extend([
        _standalone_segment("已开始查询天气数据源", call_segment_index=0, offset_sec=0.05),
        # 与工具结果同 Unit、用户时间略早：SDK 规定同 Unit 内按用户时间排序
        _standalone_segment("快好了，正在核对\n“降水概率” 🌧", call_segment_index=0,
                            offset_sec=max(response_offset - 0.01, 0.0)),
        _standalone_segment("[SUBAGENT] 结果已汇总", call_segment_index=0,
                            offset_sec=response_offset + 1.2),
    ])
    training_data = O5DuplexTrainingData.load_structure(structure)
    units, standalone_units = _assert_aligned(renderer, training_data)
    assert standalone_units >= 2, f"standalone 应落在至少 2 个 Unit，实际 {standalone_units}"
    assert units >= standalone_units


def test_standalone_with_tool_response_in_same_unit_keeps_sdk_order(renderer) -> None:
    """同一个 Unit 里 standalone 与 tool response 交错时，顺序由用户时间决定且两边一致。"""

    structure = copy.deepcopy(load_case_structure("case_3_01"))
    segments = structure["tracks"]["input_event"]["segments"]
    response_offset = segments[0]["start_trigger"]["refs"][0]["offset_sec"]
    segments.append(
        _standalone_segment("在工具结果之后同一秒内到达", call_segment_index=0,
                            offset_sec=response_offset + 0.01)
    )
    training_data = O5DuplexTrainingData.load_structure(structure)
    result = training_data.tokenize(tokenizer_id=O5TokenizerID.O5)
    events = _runtime_events_by_unit(result.arrangement)
    mixed = [e for e in events.values()
             if {"tool_response", "standalone_event"} <= {x["type"] for x in e}]
    assert mixed, "构造失败：没有出现 tool response 与 standalone 同 Unit 的情况"
    assert [x["type"] for x in mixed[0]][-2:] == ["tool_response", "standalone_event"]
    _assert_aligned(renderer, training_data)


@pytest.mark.skipif(not SUB_AGENT_BUNDLE.is_dir(), reason="本机没有 sub_agent 训练数据")
def test_standalone_tokens_match_sdk_on_real_sub_agent_samples(renderer) -> None:
    """真实数据：sub_agent 样本里 delegate_task → 若干 standalone 进度 → tool_response。"""

    checked = 0
    with (SUB_AGENT_BUNDLE / "traindata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if '"standalone"' not in line:
                continue
            training_data = O5DuplexTrainingData.load_structure(
                json.loads(line), data_root=SUB_AGENT_BUNDLE
            )
            _, standalone_units = _assert_aligned(renderer, training_data)
            assert standalone_units >= 1
            checked += 1
            if checked >= 20:
                break
    assert checked == 20
