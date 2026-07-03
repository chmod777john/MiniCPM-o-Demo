"""Minimal FC duplex runtime for the backend protocol server.

This module adapts the standalone audio_duplex_board scheduling pattern to the
backend protocol without importing the standalone server/web stack.  It keeps the
model-facing FC primitive calls intact and only owns API event shaping plus the
external-tool-id to internal-tool-id mapping.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from audio_duplex_board.tools.display_object_on_board.service import (
    DisplayObjectOnBoardService,
    board_image_result_from_tool_result,
)
from core.schemas.fc_duplex import FcToolResponse, NonSpokenStepGenerationFlag
from py_backend.media import decode_frame_base64_list


SendEvent = Callable[[str], Awaitable[None]]


DEFAULT_DISPLAY_OBJECT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "display_object_on_board",
        "description": (
            "Display a named concrete object on the visual board so the user can "
            "see it. Use only for concrete, visualizable objects mentioned in "
            "user speech."
        ),
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}


class FcDuplexSessionRuntime:
    """Small per-session scheduler for FC duplex protocol input/output."""

    def __init__(
        self,
        *,
        session_id: str,
        backend: Any,
        send: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        self.session_id = session_id
        self.backend = backend
        self._send = send
        self._response_id: Optional[str] = None
        self._tools: List[Dict[str, Any]] = []
        self._pending_tool_responses: List[FcToolResponse] = []
        self._api_to_internal: Dict[str, str] = {}
        self._internal_to_api: Dict[str, str] = {}
        self._tool_seq = 0
        self._max_spoken_tokens = 24
        self._non_spoken_budget_per_unit = 12
        self._decode_mode = "greedy"
        self._sample_rate = 16000
        self._auto_execute_tools = True
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._tool_service: Optional[DisplayObjectOnBoardService] = None

    async def prepare(self, params: Dict[str, Any]) -> None:
        config = _first_dict(params.get("config"), params.get("duplex"), params.get("fc_duplex"))
        self._max_spoken_tokens = int(config.get("max_spoken_tokens", params.get("max_spoken_tokens", 24)) or 24)
        self._non_spoken_budget_per_unit = int(
            config.get("non_spoken_budget_per_unit", params.get("non_spoken_budget_per_unit", 12)) or 12
        )
        self._decode_mode = str(config.get("decode_mode", params.get("decode_mode", "greedy")) or "greedy")
        self._sample_rate = int(config.get("sample_rate", params.get("sample_rate", 16000)) or 16000)
        self._auto_execute_tools = bool(config.get("auto_execute_tools", params.get("auto_execute_tools", True)))
        self._tools = list(params.get("tools") or [DEFAULT_DISPLAY_OBJECT_TOOL])

        tool_dir = params.get("tool_download_dir") or config.get("tool_download_dir")
        download_dir = Path(tool_dir) if tool_dir else None
        self._tool_service = DisplayObjectOnBoardService(download_dir=download_dir)

        voice = _first_dict(params.get("voice"), params.get("defaults"))
        ref_audio_path = _coalesce(params.get("ref_audio_path"), voice.get("ref_audio_path"))
        prompt_wav_path = _coalesce(params.get("prompt_wav_path"), params.get("tts_ref_audio_path"), voice.get("tts_ref_audio_path"), ref_audio_path)
        await asyncio.to_thread(
            self.backend.fc_duplex_prepare,
            system_prompt=str(_coalesce(params.get("system_prompt"), params.get("instructions"), default="")),
            tools=self._tools,
            ref_audio_path=ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            generate_audio=bool(params.get("generate_audio", True)),
        )

    async def push(self, payload: Dict[str, Any]) -> None:
        payload_type = str(payload.get("type") or payload.get("event_type") or "")
        if payload_type == "tool_result":
            await self.queue_tool_result(payload)
            return
        if payload_type in {"tool_result.delta", "tool_result.done"}:
            raise RuntimeError(f"FC runtime does not support streaming tool results yet: {payload_type}")
        await self.process_audio_input(payload)

    async def queue_tool_result(self, payload: Dict[str, Any]) -> None:
        api_id = str(payload.get("tool_call_id") or "")
        if not api_id:
            raise RuntimeError("input.tool_result requires tool_call_id")
        internal_id = self._api_to_internal.get(api_id)
        if not internal_id:
            raise RuntimeError(f"unknown tool_call_id: {api_id}")
        self._pending_tool_responses.append(
            FcToolResponse(call_id=internal_id, content=_contents_to_text(payload.get("contents")))
        )

    async def process_audio_input(self, payload: Dict[str, Any]) -> None:
        audio_base64 = _extract_audio_base64(payload)
        if not audio_base64:
            raise RuntimeError("fc_duplex input requires audio")
        input_id = payload.get("input_id")
        self._response_id = self._response_id or str(payload.get("response_id") or f"resp_{uuid.uuid4().hex[:12]}")
        frame_list = decode_frame_base64_list(_extract_frame_base64_list(payload)).frame_list
        tool_responses = list(self._pending_tool_responses)
        self._pending_tool_responses.clear()

        await asyncio.to_thread(
            self.backend.fc_duplex_prefill,
            audio_data=audio_base64,
            frame_list=frame_list,
            tool_responses=tool_responses or None,
            sample_rate=int(payload.get("sample_rate") or self._sample_rate),
        )

        spoken = await asyncio.to_thread(
            self.backend.fc_duplex_spoken_generate,
            max_tokens=self._max_spoken_tokens,
            decode_mode=self._decode_mode,
        )
        await self._emit_spoken(spoken, input_id=input_id)

        await self._run_non_spoken_loop(input_id=input_id)

        await asyncio.to_thread(self.backend.fc_duplex_finalize)

    async def close(self) -> None:
        tasks = list(self._tool_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        await asyncio.to_thread(self.backend.fc_duplex_cleanup)

    async def _run_non_spoken_loop(self, *, input_id: Optional[str]) -> None:
        steps: List[Any] = []
        for _ in range(max(0, self._non_spoken_budget_per_unit)):
            step = await asyncio.to_thread(
                self.backend.fc_duplex_non_spoken_generate,
                max_tokens=1,
                decode_mode=self._decode_mode,
            )
            steps.append(step)
            raw_flag = getattr(step, "generation_flag", "") or ""
            flag = str(getattr(raw_flag, "value", raw_flag))
            terminated = bool(getattr(step, "terminated", False))
            if terminated or flag in {
                NonSpokenStepGenerationFlag.no_action.value,
                NonSpokenStepGenerationFlag.non_spoken_slot_eos.value,
            }:
                await self._emit_non_spoken_batch(steps, input_id=input_id)
                return
        step = await asyncio.to_thread(
            self.backend.fc_duplex_non_spoken_generate,
            max_tokens=0,
            decode_mode=self._decode_mode,
            close_reason="budget_reached",
        )
        steps.append(step)
        await self._emit_non_spoken_batch(steps, input_id=input_id)

    async def _emit_spoken(self, spoken: Any, *, input_id: Optional[str]) -> None:
        is_listen = bool(getattr(spoken, "is_listen", False))
        is_speaking = bool(getattr(spoken, "is_speaking", False))
        text = str(getattr(spoken, "spoken_text", "") or "")
        waveform = getattr(spoken, "audio_waveform", None)
        metadata = _model_to_dict(spoken)
        metadata.pop("audio_waveform", None)

        if is_listen:
            await self._send_sp_token("listen", input_id=input_id)
            await self._send(
                "response.output.delta",
                kind="listen",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                metrics=metadata,
            )
            return
        if is_speaking:
            await self._send_sp_token("speak", input_id=input_id)
        if text:
            await self._send(
                "response.output.delta",
                kind="text",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                text=text,
                metrics=metadata,
            )
        if is_speaking and waveform is not None:
            await self._send(
                "response.output.delta",
                kind="audio",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                audio=_audio_waveform_to_float32_base64(waveform),
                sample_rate=int(getattr(spoken, "audio_sample_rate", None) or 24000),
                metrics=metadata,
            )
        if bool(getattr(spoken, "spoken_turn_eos", False)):
            await self._send_sp_token("spoken_turn_eos", input_id=input_id)

    async def _emit_non_spoken_batch(self, steps: List[Any], *, input_id: Optional[str]) -> None:
        token_strs: List[str] = []
        text_parts: List[str] = []
        close_reason: Optional[str] = None
        closed_spans: List[Any] = []

        for step in steps:
            token_strs.extend(str(token) for token in list(getattr(step, "token_strs", None) or []))
            text = str(getattr(step, "text", "") or "")
            if text:
                text_parts.append(text)
            step_close_reason = getattr(step, "close_reason", None)
            if step_close_reason:
                close_reason = str(step_close_reason)
            closed_spans.extend(list(getattr(step, "closed_spans", None) or []))

        text = "".join(text_parts)
        if token_strs or text:
            await self._send(
                "response.output.delta",
                kind="non_spoken",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                text=text,
                token_strs=token_strs,
            )
        if close_reason:
            token = _non_spoken_close_reason_to_sp_token(close_reason)
            if token:
                await self._send_sp_token(token, input_id=input_id)
        for span in closed_spans:
            await self._emit_closed_span(span, input_id=input_id)

    async def _send_sp_token(self, token: str, *, input_id: Optional[str]) -> None:
        await self._send(
            "response.output.sp_tokens",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            token=token,
        )

    async def _emit_closed_span(self, span: Any, *, input_id: Optional[str]) -> None:
        span_type = getattr(span, "type", None)
        if span_type == "think":
            await self._send("response.think.begin", session_id=self.session_id, response_id=self._response_id, input_id=input_id)
            text = str(getattr(span, "text", "") or "")
            if text:
                await self._send(
                    "response.think.delta",
                    session_id=self.session_id,
                    response_id=self._response_id,
                    input_id=input_id,
                    delta=text,
                )
            await self._send("response.think.end", session_id=self.session_id, response_id=self._response_id, input_id=input_id)
            return
        if span_type != "tool_call":
            return

        internal_id = getattr(span, "tool_call_id", None)
        api_id = self._api_id_for_internal(str(internal_id) if internal_id else None)
        wire = getattr(span, "wire", None) or ""
        await self._send(
            "response.tool_call.args.begin",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            tool_call_id=api_id,
        )
        if wire:
            await self._send(
                "response.tool_call.args.delta",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                tool_call_id=api_id,
                delta=wire,
            )
        await self._send(
            "response.tool_call.args.end",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            tool_call_id=api_id,
        )
        raw = _tool_call_raw(span)
        await self._send(
            "response.tool_call.args.raw",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            tool_call_id=api_id,
            raw=raw,
        )
        if self._auto_execute_tools and not raw.get("error"):
            task = asyncio.create_task(self._auto_execute_tool(api_id=api_id, raw=raw))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def _auto_execute_tool(self, *, api_id: str, raw: Dict[str, Any]) -> None:
        name = str(raw.get("name") or "")
        if name != "display_object_on_board" or self._tool_service is None:
            return
        args = _parse_arguments(raw.get("arguments"))
        query = str(args.get("name") or "").strip()
        result = await asyncio.to_thread(self._tool_service.search, query)
        image = board_image_result_from_tool_result(result, tool_call_id=api_id)
        await self._send(
            "response.tool_result",
            session_id=self.session_id,
            response_id=self._response_id,
            tool_call_id=api_id,
            name=name,
            result={"query": result.query, "image": _model_to_dict(image), "error": result.error},
        )
        internal_id = self._api_to_internal.get(api_id)
        if internal_id:
            self._pending_tool_responses.append(
                FcToolResponse(call_id=internal_id, content=result.tool_response_content)
            )

    def _api_id_for_internal(self, internal_id: Optional[str]) -> str:
        if internal_id and internal_id in self._internal_to_api:
            return self._internal_to_api[internal_id]
        self._tool_seq += 1
        api_id = f"tc_{self._tool_seq:06d}"
        if internal_id:
            self._internal_to_api[internal_id] = api_id
            self._api_to_internal[api_id] = internal_id
        return api_id


def fc_duplex_enabled(params: Dict[str, Any]) -> bool:
    config = _first_dict(params.get("config"), params.get("duplex"), params.get("fc_duplex"))
    value = _coalesce(params.get("fc_duplex"), params.get("runtime"), config.get("runtime"), config.get("enabled"))
    if isinstance(value, str):
        return value in {"fc", "fc_duplex", "true", "1", "yes"}
    return bool(value)


def _non_spoken_close_reason_to_sp_token(reason: str) -> Optional[str]:
    if reason == "eos":
        return "non_spoken_eos"
    if reason == "no_action":
        return "no_action"
    if reason == "budget_reached":
        return "non_spoken_budget_reached"
    if reason == "hold":
        return "non_spoken_hold"
    if reason == "abort":
        return "non_spoken_abort"
    return None


def _tool_call_raw(span: Any) -> Dict[str, Any]:
    error = getattr(span, "error", None)
    tool_call = getattr(span, "tool_call", None)
    if error:
        return {"error": str(error)}
    if not isinstance(tool_call, dict):
        return {"error": "missing parsed tool call"}
    name = tool_call.get("name")
    if not name:
        return {"error": "missing tool call name"}
    return {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False),
    }


def _parse_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _contents_to_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if not isinstance(contents, list):
        return json.dumps(contents, ensure_ascii=False)
    parts: List[str] = []
    for item in contents:
        if isinstance(item, dict):
            if item.get("kind") == "text" or "text" in item:
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return "".join(parts)


def _audio_waveform_to_float32_base64(audio_waveform: Any) -> str:
    array = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
    return base64.b64encode(array.tobytes()).decode("utf-8")


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _extract_frame_base64_list(payload: Dict[str, Any]) -> Optional[list[str]]:
    direct = payload.get("frame_base64_list") or payload.get("video_frames")
    if direct:
        return list(direct)
    frames = payload.get("frames")
    if not frames:
        return None
    out: List[str] = []
    for frame in frames:
        if isinstance(frame, str):
            out.append(frame)
        elif isinstance(frame, dict):
            data = frame.get("data") or frame.get("base64")
            if data:
                out.append(data)
    return out or None


def _extract_audio_base64(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("audio_base64", "audio_data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    audio = payload.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, dict):
        value = audio.get("data") or audio.get("base64") or audio.get("audio_base64")
        if isinstance(value, str) and value:
            return value
    return None
