"""按训练样本回放一次 API 会话，验收模型对 standalone input_event 的反应。

用 sub_agent 训练样本驱动锁步客户端：口径与训练一致（data-mix 的 system_prefix_text 原样插到
system 最前，样本自带的参考音频 / 工具 / UnitPolicy），用户音频逐 Unit 原样喂入。

事件时刻按模型的**实际**行为对齐，而不是照抄样本的绝对 Unit：
- tool response 与发起子任务之后的 standalone 进度，挂在模型第 k 次工具调用上，
  保持与样本中第 k 次调用相同的 Unit 间隔；模型没有发出第 k 次调用时，这些事件不发并记录下来。
- 第一次工具调用之前的 standalone（如 ``[FILE_UPLOADED] ...`` 这类用户侧通知）按样本原 Unit 发。

输出每条样本的逐 Unit 记录（JSONL）和一份「模型 vs 训练样本」并排对照（Markdown）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clients.o5_fc_lockstep_client import (  # noqa: E402
    O5FcLockstepSession,
    O5FcSessionError,
    O5FcSessionSpec,
    O5FcStandaloneEvent,
    O5FcToolResult,
)
from minicpm_o5_sdk import O5DuplexTrainingData, O5TokenizerID  # noqa: E402
from minicpm_o5_sdk.protocols.duplex.training_data import O5SystemTextSegment  # noqa: E402
from minicpm_o5_sdk.protocols.duplex.training_data.events import (  # noqa: E402
    O5StandaloneEvent,
    O5ToolCreatedEvent,
    O5ToolResponseEvent,
)

SAMPLES_PER_UNIT = 16_000
TAIL_UNITS = 10
"""最后一个事件发出后再跑的静音 Unit 数，给模型留出转述最终结果的时间。"""


@dataclass
class ScheduledEvent:
    """一个待发的样本事件。anchor_call 为 None 表示按样本原 Unit 发。"""

    kind: str  # "standalone" | "tool_response"
    text: str
    gt_unit: int
    anchor_call: Optional[int] = None
    delta_units: int = 0
    event_id: Optional[str] = None
    sent_unit: Optional[int] = None


@dataclass
class GtCall:
    """样本里的第 k 次工具调用。"""

    tool_call_id: str
    name: str
    close_unit: int


@dataclass
class SampleRecord:
    """一条样本的回放结果。"""

    line_index: int
    data_id: Optional[str]
    units: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    unsent_events: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _load_sample(line: str, root: Path, prefix: Optional[str]) -> O5DuplexTrainingData:
    """加载样本并按训练代码（item_builder → prepend_system_prefix_text）插入前缀。"""

    training_data = O5DuplexTrainingData.load_structure(json.loads(line), data_root=root)
    if prefix:
        training_data.system.segments.insert(0, O5SystemTextSegment(text=prefix))
    return training_data


def _session_system(training_data: O5DuplexTrainingData, root: Path) -> dict[str, Any]:
    """把样本 system 转成 session.init 的 system（音频改为服务端可读的绝对路径）。"""

    segments: list[dict[str, Any]] = []
    for segment in training_data.system.segments:
        kind = getattr(segment, "kind", None)
        if kind == "text":
            segments.append({"kind": "text", "text": segment.text})
        elif kind == "audio":
            raw = getattr(segment.audio, "file_path", None)
            if not raw:
                raise ValueError("system audio 段缺 file_path，无法回放")
            path = Path(raw)
            path = path if path.is_absolute() else root / path
            segments.append({"kind": "audio", "audio": {"source": "path", "file_path": str(path)}})
        else:
            raise ValueError(f"未支持的 system 段类型: {kind!r}")
    tools = [tool.model_dump(mode="json", exclude_none=True) for tool in training_data.system.tools]
    return {"segments": segments, "tools": tools}


def _gt_schedule(training_data: O5DuplexTrainingData) -> tuple[list[GtCall], list[ScheduledEvent], int]:
    """从 SDK arrangement 读出样本的工具调用与 input event 调度（不从秒值反推）。"""

    arrangement = training_data.tokenize(tokenizer_id=O5TokenizerID.O5).arrangement
    created_unit: dict[str, int] = {}
    responses: list[tuple[int, str, str]] = []
    standalones: list[tuple[int, str]] = []
    for segment in arrangement.tracks.input_event.segments:
        event = segment.event
        unit = segment.perceived_input_gate.unit_index
        if isinstance(event, O5ToolCreatedEvent):
            created_unit[event.tool_call_id] = unit
        elif isinstance(event, O5ToolResponseEvent):
            responses.append((unit, event.tool_call_id, event.contents[0].text))
        elif isinstance(event, O5StandaloneEvent):
            standalones.append((unit, event.contents[0].text))
    calls: list[GtCall] = []
    for segment in arrangement.tracks.ai_non_spoken.segments:
        content = segment.content
        if getattr(content, "kind", None) == "tool_call":
            # created 固定落在调用闭合 Unit 的下一个（SDK tool-protocol §1.1）
            calls.append(GtCall(content.tool_call_id, content.name,
                                created_unit[content.tool_call_id] - 1))
    calls.sort(key=lambda call: call.close_unit)
    call_index = {call.tool_call_id: k for k, call in enumerate(calls)}

    events: list[ScheduledEvent] = []
    for unit, call_id, text in responses:
        k = call_index[call_id]
        events.append(ScheduledEvent("tool_response", text, unit, k, unit - calls[k].close_unit))
    for n, (unit, text) in enumerate(standalones):
        earlier = [k for k, call in enumerate(calls) if call.close_unit < unit]
        anchor = earlier[-1] if earlier else None
        delta = unit - calls[anchor].close_unit if anchor is not None else 0
        events.append(ScheduledEvent("standalone", text, unit, anchor, delta, event_id=f"sa_{n}"))
    events.sort(key=lambda e: (e.gt_unit, e.kind != "tool_response"))
    return calls, events, len(arrangement.unit_spans) if hasattr(arrangement, "unit_spans") else 0


def _gt_timeline(training_data: O5DuplexTrainingData) -> list[tuple[int, str, str]]:
    """样本里的参考行为：(unit, 类别, 文本)。"""

    arrangement = training_data.tokenize(tokenizer_id=O5TokenizerID.O5).arrangement
    rows: list[tuple[int, str, str]] = []
    for segment in arrangement.tracks.input_event.segments:
        event = segment.event
        if isinstance(event, (O5StandaloneEvent, O5ToolResponseEvent)):
            label = "standalone" if isinstance(event, O5StandaloneEvent) else "tool_response"
            rows.append((segment.perceived_input_gate.unit_index, f"EVENT {label}", event.contents[0].text))
    for segment in arrangement.tracks.ai_spoken.segments:
        rows.append((segment.timeline.start_unit_index, "SPEAK", segment.text))
    for segment in arrangement.tracks.ai_non_spoken.segments:
        content = segment.content
        if getattr(content, "kind", None) == "tool_call":
            rows.append((segment.timeline.start_unit_index, "TOOL_CALL",
                         f"{content.name} {json.dumps(content.arguments, ensure_ascii=False)}"))
        elif getattr(content, "text", None):
            rows.append((segment.timeline.start_unit_index, "THINK", content.text))
    return sorted(rows, key=lambda row: row[0])


def replay_sample(
    *, server_url: str, line_index: int, line: str, root: Path, prefix: Optional[str],
    tts_pad_protocol: str, decoding_policy: str,
) -> tuple[SampleRecord, list[tuple[int, str, str]]]:
    """回放一条样本，返回逐 Unit 记录与样本参考时间线。"""

    training_data = _load_sample(line, root, prefix)
    record = SampleRecord(line_index=line_index, data_id=getattr(training_data, "data_id", None))
    gt_calls, events, _ = _gt_schedule(training_data)
    gt_rows = _gt_timeline(training_data)
    audio = training_data.build_user_audio_tensor(tokenizer_id=O5TokenizerID.O5)
    audio_units = audio.units.numpy().astype(np.float32) if audio is not None else np.zeros((0, SAMPLES_PER_UNIT), np.float32)

    spec = O5FcSessionSpec(
        server_url=server_url,
        system=_session_system(training_data, root),
        unit_policy=training_data.unit_policy.model_dump(mode="json"),
        tts_prompt_audio_path=None,  # 验收只看文本；不出音频省时间
        tts_pad_protocol=tts_pad_protocol,
        decoding_policy=decoding_policy,
    )
    model_call_units: list[int] = []
    model_call_ids: list[str] = []
    session = O5FcLockstepSession(spec)
    try:
        session.open()
        unit = 0
        while True:
            pending = [e for e in events if e.sent_unit is None]
            due: list[ScheduledEvent] = []
            for event in pending:
                if event.anchor_call is None:
                    if event.gt_unit <= unit:
                        due.append(event)
                elif event.anchor_call < len(model_call_units):
                    if model_call_units[event.anchor_call] + event.delta_units <= unit:
                        due.append(event)
            payload = []
            for event in due:
                event.sent_unit = unit
                if event.kind == "standalone":
                    payload.append(O5FcStandaloneEvent(event.text, event_id=event.event_id))
                else:
                    payload.append(O5FcToolResult(model_call_ids[event.anchor_call], event.text))
            pcm = audio_units[unit] if unit < len(audio_units) else np.zeros(SAMPLES_PER_UNIT, np.float32)
            started = time.time()
            result = session.step(pcm, input_events=payload)
            for call in result.tool_calls:
                model_call_units.append(unit)
                model_call_ids.append(call.tool_call_id)
                record.model_calls.append({"unit": unit, "name": call.name, "arguments": call.arguments})
            record.units.append({
                "unit": unit,
                "sent": [{"kind": e.kind, "text": e.text, "event_id": e.event_id} for e in due],
                "consumed_standalone": result.consumed_standalone_event_ids,
                "consumed_tool": result.consumed_tool_call_ids,
                "spoken": result.spoken_text,
                "think": result.think_text,
                "tool_calls": [c.name for c in result.tool_calls],
                "wall_sec": round(time.time() - started, 3),
            })
            unit += 1
            still_pending = [
                e for e in events if e.sent_unit is None
                and (e.anchor_call is None or e.anchor_call < len(model_call_units))
            ]
            waiting_on_call = [
                e for e in events if e.sent_unit is None and e.anchor_call is not None
                and e.anchor_call >= len(model_call_units)
            ]
            last_sent = max((e.sent_unit for e in events if e.sent_unit is not None), default=-1)
            audio_done = unit >= len(audio_units)
            if audio_done and not still_pending and (not waiting_on_call or unit > len(audio_units) + 30) \
                    and unit > last_sent + TAIL_UNITS:
                break
    except O5FcSessionError as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 关会话失败不影响已拿到的记录
            pass
    record.unsent_events = [
        {"kind": e.kind, "text": e.text, "gt_unit": e.gt_unit, "anchor_call": e.anchor_call}
        for e in events if e.sent_unit is None
    ]
    return record, gt_rows


def _speak_after(record: SampleRecord, unit: int, window: int) -> Optional[tuple[int, str]]:
    for row in record.units:
        if unit <= row["unit"] < unit + window and row["spoken"].strip():
            return row["unit"], row["spoken"].strip()
    return None


def render_markdown(record: SampleRecord, gt_rows: list[tuple[int, str, str]]) -> str:
    """并排列出模型行为与样本参考行为，并对每条 standalone 给出「3 Unit 内是否开口」。"""

    lines = [f"## 样本第 {record.line_index} 行 `{record.data_id}`", ""]
    if record.error:
        lines += [f"**会话出错**：`{record.error}`", ""]
    lines += ["### 模型（本次回放）", "", "| Unit | 注入事件 | 模型说 | 模型 think / 工具 |", "|---:|---|---|---|"]
    for row in record.units:
        sent = "<br>".join(f"[{e['kind']}] {e['text'][:60]}" for e in row["sent"])
        extra = " ".join(filter(None, [row["think"][:60], " ".join(f"→{n}" for n in row["tool_calls"])]))
        if sent or row["spoken"].strip() or extra:
            lines.append(f"| {row['unit']} | {sent} | {row['spoken'].strip()[:80]} | {extra} |")
    lines += ["", "### 训练样本（参考）", "", "| Unit | 类别 | 内容 |", "|---:|---|---|"]
    for unit, kind, text in gt_rows:
        lines.append(f"| {unit} | {kind} | {text[:90].replace(chr(10), ' ')} |")
    lines += ["", "### standalone 注入后 3 个 Unit 内模型是否开口", ""]
    for row in record.units:
        for event in row["sent"]:
            if event["kind"] != "standalone":
                continue
            hit = _speak_after(record, row["unit"], 3)
            verdict = f"✅ U{hit[0]}「{hit[1][:50]}」" if hit else "— 没有开口"
            lines.append(f"- U{row['unit']} 「{event['text'][:50]}」 → {verdict}")
    if record.unsent_events:
        lines += ["", f"未发出的事件（模型没有发起对应的工具调用）：{len(record.unsent_events)} 条"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--sources", required=True, help="data-mix 各源的 root / system_prefix_text（JSON）")
    parser.add_argument("--source", required=True, action="append", help="source_id，可重复")
    parser.add_argument("--samples", type=int, default=3, help="每个源回放几条含 standalone 的样本")
    parser.add_argument("--tts-pad-protocol", required=True, choices=["legacy", "revised"])
    parser.add_argument("--decoding-policy", default="raise_protocol_error",
                        choices=["raise_protocol_error", "constrained_decoding"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for source_id in args.source:
        meta = sources[source_id]
        root = Path(meta["root"])
        report = [f"# {source_id}", "", f"数据：`{root}`", ""]
        picked = 0
        with (root / "traindata.jsonl").open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if '"standalone"' not in line:
                    continue
                record, gt_rows = replay_sample(
                    server_url=args.server_url, line_index=line_index, line=line, root=root,
                    prefix=meta.get("system_prefix_text"), tts_pad_protocol=args.tts_pad_protocol,
                    decoding_policy=args.decoding_policy,
                )
                with (out / f"{source_id}.jsonl").open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
                report.append(render_markdown(record, gt_rows))
                (out / f"{source_id}.md").write_text("\n".join(report), encoding="utf-8")
                print(f"[replay] {source_id} 第 {line_index} 行：{len(record.units)} Unit"
                      f"{'，出错 ' + record.error if record.error else ''}", flush=True)
                picked += 1
                if picked >= args.samples:
                    break


if __name__ == "__main__":
    main()
