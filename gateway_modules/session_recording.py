"""Gateway-side session recorder (faithful stream capture).

录音机不是剪辑师:gateway 在转发通路上看到什么协议帧,就忠实记什么——不聚合、不配对、
不拼连续 audio、不判轮次。复盘时由消费方(前端)按需重构。

落盘结构 `data/sessions/<session_id>/`:
  - meta.json      会话级元信息(收尾补全 ended_at/duration/close_reason)
  - stream.jsonl   忠实事件流,每行一个帧 {seq, ts, dir, frame};frame 为协议帧原样
  - blob/NNN.ext   大块 base64 二进制外置(audio→.wav+.f32, jpeg→.jpg),stream.jsonl 里用 "@blob/NNN.ext" 指针引用

设计要点:
  - JSONL append-only,边录边写,进程崩溃不丢已录部分,不在内存攒整会话。
  - 二进制解码 + 落盘 + 行追加都走线程池,绝不阻塞 gateway 转发协程。
  - 全部对外方法 fail-safe:录制是旁路,任何异常都不应影响转发主路径(由调用方 try/except 兜)。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np

# 上行用户音频 16k / 下行模型音频 24k(schema §1.3 实现现状)
_USER_AUDIO_SR = 16000
_AI_AUDIO_SR = 24000

# 录制 I/O 专用线程池,与转发事件循环解耦
_io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rec_io")


def _write_wav(path: str, pcm_float32: np.ndarray, sample_rate: int) -> None:
    """float32 mono → 16-bit PCM WAV(手写 RIFF,只依赖 numpy)。"""
    pcm16 = np.clip(pcm_float32 * 32767, -32768, 32767).astype(np.int16)
    n_channels, sampwidth = 1, 2
    data_size = len(pcm16) * sampwidth
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                            sample_rate * n_channels * sampwidth,
                            n_channels * sampwidth, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm16.tobytes())


def _audio_payload_meta(raw: bytes, *, sample_rate: int, wav_rel: str, f32_rel: str) -> Dict[str, Any]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nbytes": len(raw),
        "samples": len(raw) // 4,
        "dtype": "float32-le",
        "sample_rate": sample_rate,
        "blob_f32": f32_rel,
        "blob_wav": wav_rel,
    }


def _jpeg_payload_meta(raw: bytes, *, jpg_rel: str) -> Dict[str, Any]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nbytes": len(raw),
        "blob_jpg": jpg_rel,
    }


class SessionRecorder:
    """忠实录音机:每帧原样存,二进制外置。不解析 kind、不聚合、不分轨。"""

    def __init__(
        self,
        session_id: str,
        mode: str,
        *,
        data_dir: str,
        identity: Optional[Dict[str, Any]] = None,
        worker: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.mode = mode
        self._dir = os.path.join(data_dir, "sessions", session_id)
        self._blob_dir = os.path.join(self._dir, "blob")
        self._stream_path = os.path.join(self._dir, "stream.jsonl")
        self._meta_path = os.path.join(self._dir, "meta.json")

        self._seq = 0
        self._blob_idx = 0
        self._lock = threading.Lock()       # 保护 seq/blob_idx/jsonl 追加顺序
        self._start = time.time()
        self._closed = False

        os.makedirs(self._blob_dir, exist_ok=True)
        self._meta: Dict[str, Any] = {
            "session_id": session_id,
            "mode": mode,
            "created_at": self._start,
            "ended_at": None,
            "duration_s": None,
            "worker": worker or {},
            # identity 直接整块存:client_id / page_session_id / page_route /
            # client_surface / client_ip / user_agent / source_* 等,前端新增字段自动带入。
            "identity": identity or {},
            "close_reason": None,
        }
        self._flush_meta()

    # ---------- 对外:录一帧(fail-safe 由调用方包裹,但这里也吞自身异常) ----------

    def record(self, direction: str, frame: Dict[str, Any]) -> None:
        """记录一个协议帧。direction = "up"(client→backend) | "down"(backend→client)。"""
        ts = time.time()
        try:
            with self._lock:
                seq = self._seq
                self._seq += 1
                # 深拷贝 + 二进制外置(在锁内分配 blob_idx,保证编号单调)
                externalized, payload_trace = self._externalize(frame)
            event = {"seq": seq, "ts": ts, "dir": direction, "frame": externalized}
            if payload_trace is not None:
                event["payload_trace"] = payload_trace
            line = json.dumps(event, ensure_ascii=False)
            _io_pool.submit(self._append_line, line)
        except Exception:
            # 录制是旁路,绝不影响转发
            pass

    def close(self, reason: Optional[str] = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._meta["ended_at"] = time.time()
        self._meta["duration_s"] = round(self._meta["ended_at"] - self._start, 1)
        self._meta["close_reason"] = reason
        _io_pool.submit(self._flush_meta)

    # ---------- 内部 ----------

    def _next_blob(self, ext: str) -> str:
        """分配一个 blob 文件名(调用方需持锁)。返回 (相对指针, 绝对路径)。"""
        rel = f"@blob/{self._blob_idx:04d}.{ext}"
        self._blob_idx += 1
        return rel

    def _next_blob_stem(self) -> str:
        stem = f"@blob/{self._blob_idx:04d}"
        self._blob_idx += 1
        return stem

    def _externalize(self, frame: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """把帧里的大块 base64 二进制抽到 blob/,字段值换成 @blob 指针。返回新 dict(不改原帧)。"""
        ftype = frame.get("type")

        # 上行 input.append:input.audio(f32) + input.video_frames(jpeg[])
        if ftype == "input.append" and isinstance(frame.get("input"), dict):
            inp = dict(frame["input"])
            payload_trace: Dict[str, Any] = {
                "force_listen": bool(inp.get("force_listen", False)),
                "max_slice_nums": int(inp.get("max_slice_nums", 1) or 1),
            }
            audio = inp.get("audio")
            if isinstance(audio, str) and audio:
                audio_meta = self._stash_audio(audio, _USER_AUDIO_SR)
                inp["audio"] = audio_meta["blob_wav"]
                payload_trace["audio"] = audio_meta
            frames = inp.get("video_frames")
            if isinstance(frames, list) and frames:
                out_frames = []
                frame_meta = []
                for fr in frames:
                    if isinstance(fr, str) and fr:
                        meta = self._stash_jpeg(fr)
                        out_frames.append(meta["blob_jpg"])
                        frame_meta.append(meta)
                    else:
                        out_frames.append(fr)
                        frame_meta.append(None)
                inp["video_frames"] = out_frames
                payload_trace["video_frames"] = frame_meta
            return {**frame, "input": inp}, payload_trace

        # 下行 response.output.delta kind=audio:audio(f32)
        if ftype == "response.output.delta" and frame.get("kind") == "audio":
            audio = frame.get("audio")
            if isinstance(audio, str) and audio:
                audio_meta = self._stash_audio(audio, _AI_AUDIO_SR)
                return {**frame, "audio": audio_meta["blob_wav"]}, {"audio": audio_meta}

        # response.done 可能带 audio(非流式)
        if ftype == "response.done":
            audio = frame.get("audio")
            if isinstance(audio, str) and audio:
                audio_meta = self._stash_audio(audio, _AI_AUDIO_SR)
                return {**frame, "audio": audio_meta["blob_wav"]}, {"audio": audio_meta}

        # 其它帧(session.init/created/text/listen/closed 等)原样
        init_trace = self._session_init_trace(frame) if ftype == "session.init" else None
        return frame, init_trace

    @staticmethod
    def _session_init_trace(frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return None
        trace: Dict[str, Any] = {}
        for key in ("ref_audio_base64", "tts_ref_audio_base64"):
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                continue
            try:
                raw = base64.b64decode(value)
                trace[key] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "nbytes": len(raw),
                    "samples": len(raw) // 4,
                    "dtype": "float32-le",
                    "sample_rate": _USER_AUDIO_SR,
                }
            except Exception:
                trace[key] = {"decode_error": True}
        return trace or None

    def _stash_audio(self, b64: str, sr: int) -> Dict[str, Any]:
        stem = self._next_blob_stem()  # 持锁中调用
        wav_rel = f"{stem}.wav"
        f32_rel = f"{stem}.f32"
        wav_path = os.path.join(self._dir, wav_rel[1:])  # 去掉前导 '@'
        f32_path = os.path.join(self._dir, f32_rel[1:])
        raw = base64.b64decode(b64)
        _io_pool.submit(self._write_audio_blob, wav_path, f32_path, raw, sr)
        return _audio_payload_meta(raw, sample_rate=sr, wav_rel=wav_rel, f32_rel=f32_rel)

    def _stash_jpeg(self, b64: str) -> Dict[str, Any]:
        rel = self._next_blob("jpg")
        abs_path = os.path.join(self._dir, rel[1:])
        raw = base64.b64decode(b64)
        _io_pool.submit(self._write_jpeg_blob, abs_path, raw)
        return _jpeg_payload_meta(raw, jpg_rel=rel)

    @staticmethod
    def _write_audio_blob(wav_path: str, f32_path: str, raw: bytes, sr: int) -> None:
        try:
            with open(f32_path, "wb") as f:
                f.write(raw)
            _write_wav(wav_path, np.frombuffer(raw, dtype="<f4"), sr)
        except Exception:
            pass

    @staticmethod
    def _write_jpeg_blob(path: str, raw: bytes) -> None:
        try:
            with open(path, "wb") as f:
                f.write(raw)
        except Exception:
            pass

    def _append_line(self, line: str) -> None:
        try:
            with open(self._stream_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _flush_meta(self) -> None:
        try:
            tmp = self._meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._meta_path)
        except Exception:
            pass
