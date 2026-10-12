"""Gateway-side session recorder (faithful stream capture).

录音机不是剪辑师:gateway 在转发通路上看到什么协议帧,就忠实记什么——不聚合、不配对、
不拼连续 audio、不判轮次、不知道这个 Session 是拿来干什么的。复盘时由消费方按需重构。

落盘结构 `<sessions_dir>/<session_id>/`(默认 sessions_dir = `<data_dir>/sessions`):
  - meta.json      会话级元信息(收尾补全 ended_at/duration/close_reason)
  - stream.jsonl   忠实事件流,每行一个帧:
                     {seq, dir, src, recv_ts, send_ts, ts, frame}
                   dir     "up"(client→backend) | "down"(→client)
                   src     "client" | "worker" | "gateway"(网关自己发的排队/错误/超时帧)
                   recv_ts 网关收到这一帧的时刻(gateway 自产帧 = 产生时刻)
                   send_ts 网关把它转发/发送出去的时刻(发送失败为 null)
                   ts      = recv_ts,保留给旧读取方
                   frame   协议帧原样;worker 自己写的 server_send_ts 仍在帧里
  - blob/NNNN.wav  帧里的 base64 音频外置为 **32-bit float WAV**,stream.jsonl 里用
                   "@blob/NNNN.wav" 指针引用;blob/NNNN.jpg 为视频帧 JPEG 原字节

无损承诺(可逐字节还原原帧):
  - 音频 WAV 的 data 段 = 原 base64 字段解码后的字节,一个字节不改。
    还原:`base64.b64encode(data段字节).decode()` 即原字段值。
  - 只在「重新编码与原字符串完全相同」时才外置;否则(非标准 base64、长度不是 4 的倍数等)
    原样留在帧里。所以外置与否不影响可还原性。
  - 不按事件名枚举:帧里任意层级、键名为 ``audio`` / ``audio_base64`` 且值为字符串的字段
    都按此处理,新旧两代协议都覆盖(旧协议 `input.audio` / `response.output.delta.audio`,
    v3 的 `input.audio_base64` / `response.spoken.delta.audio`)。
    WAV 头里的采样率取同一层的 ``sample_rate``,缺省按方向(上行 16k / 下行 24k)——
    采样率只是头部元信息,不影响 data 段字节。

写入顺序:
  - 每个 recorder 一个**单线程**写入器,按 seq 顺序串行落盘(解码、写 blob、追加行、写 meta)。
    早先是全局 4 线程池并发 `open('a')` 追加同一文件:行序不保证、大行在 GPFS 上有交错风险。
  - 绝不阻塞 gateway 转发协程;全部对外方法 fail-safe(录制是旁路)。
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

_USER_AUDIO_SR = 16000
_AI_AUDIO_SR = 24000
_AUDIO_KEYS = ("audio", "audio_base64")
# 太短的不外置(大概率不是音频负载),原样留在帧里
_MIN_EXTERNALIZE_CHARS = 64


def _write_float32_wav(path: str, data: bytes, sample_rate: int) -> None:
    """把原始 float32 小端字节原样写成 WAVE_FORMAT_IEEE_FLOAT(3) 单声道 WAV。

    data 段 = 入参字节,不做任何数值变换 —— 这是无损的关键。
    """
    n_channels, sampwidth = 1, 4
    data_size = len(data)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, n_channels, sample_rate,
                            sample_rate * n_channels * sampwidth,
                            n_channels * sampwidth, 32))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def read_wav_data_bytes(path: str) -> bytes:
    """读出 WAV 的 data 段原始字节(供还原 / 校验)。"""
    with open(path, "rb") as f:
        raw = f.read()
    pos = 12
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"data":
            return raw[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
    raise ValueError(f"no data chunk: {path}")


def _strict_b64decode(value: str) -> Optional[bytes]:
    """严格解码;只有「重新编码后与原串完全相同」才返回字节,否则 None(=原样保留)。"""
    if len(value) < _MIN_EXTERNALIZE_CHARS or len(value) % 4:
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    if base64.b64encode(data).decode("ascii") != value:
        return None
    return data


class SessionRecorder:
    """忠实录音机:每帧原样存,音视频二进制无损外置。不解析业务、不聚合、不分轨。"""

    def __init__(
        self,
        session_id: str,
        mode: str,
        *,
        data_dir: Optional[str] = None,
        sessions_dir: Optional[str] = None,
        identity: Optional[Dict[str, Any]] = None,
        worker: Optional[Dict[str, Any]] = None,
    ) -> None:
        if sessions_dir is None:
            if data_dir is None:
                raise ValueError("SessionRecorder 需要 sessions_dir 或 data_dir")
            sessions_dir = os.path.join(data_dir, "sessions")
        self.session_id = session_id
        self.mode = mode
        self._dir = os.path.join(sessions_dir, session_id)
        self._blob_dir = os.path.join(self._dir, "blob")
        self._stream_path = os.path.join(self._dir, "stream.jsonl")
        self._meta_path = os.path.join(self._dir, "meta.json")

        self._seq = 0
        self._blob_idx = 0
        self._lock = threading.Lock()       # 保护 seq 分配与提交顺序
        self._start = time.time()
        self._closed = False
        self._fh = None
        # 单线程写入器:提交顺序 = 落盘顺序
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rec_{session_id[:12]}")

        os.makedirs(self._blob_dir, exist_ok=True)
        self._meta: Dict[str, Any] = {
            "session_id": session_id,
            "mode": mode,
            "created_at": self._start,
            "ended_at": None,
            "duration_s": None,
            "worker": worker or {},
            # identity 直接整块存:client_id / page_session_id / source_* 等(连接参数,非业务语义)
            "identity": identity or {},
            "close_reason": None,
            "frames": 0,
            "format": {
                "stream": "seq,dir,src,recv_ts,send_ts,ts,frame",
                "audio_blob": "float32-wav-lossless",
            },
        }
        self._writer.submit(self._flush_meta)

    # ---------- 对外 ----------

    def record(
        self,
        direction: str,
        frame: Dict[str, Any],
        *,
        recv_ts: Optional[float] = None,
        send_ts: Optional[float] = None,
        src: Optional[str] = None,
    ) -> None:
        """记录一个协议帧。调用方之后不得再修改 frame(写入在后台线程完成)。"""
        if recv_ts is None:
            recv_ts = time.time()
        if src is None:
            src = "client" if direction == "up" else "worker"
        try:
            with self._lock:
                if self._closed:
                    return
                seq = self._seq
                self._seq += 1
                self._writer.submit(self._write_frame, seq, direction, src, recv_ts, send_ts, frame)
        except Exception:
            pass

    def close(self, reason: Optional[str] = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            ended = time.time()
            self._meta["ended_at"] = ended
            self._meta["duration_s"] = round(ended - self._start, 3)
            self._meta["close_reason"] = reason
            try:
                self._writer.submit(self._finish)
                self._writer.shutdown(wait=False)
            except Exception:
                pass

    def wait_closed(self) -> None:
        """阻塞到写入线程把已提交的帧全部落盘(close 之后调用;测试/离线工具用)。"""
        self._writer.shutdown(wait=True)

    @property
    def directory(self) -> str:
        return self._dir

    # ---------- 写入线程内 ----------

    def _write_frame(self, seq: int, direction: str, src: str,
                     recv_ts: float, send_ts: Optional[float], frame: Dict[str, Any]) -> None:
        try:
            default_sr = _USER_AUDIO_SR if direction == "up" else _AI_AUDIO_SR
            externalized = self._externalize(frame, default_sr)
            line = json.dumps(
                {"seq": seq, "dir": direction, "src": src,
                 "recv_ts": recv_ts, "send_ts": send_ts, "ts": recv_ts,
                 "frame": externalized},
                ensure_ascii=False,
            )
            if self._fh is None:
                self._fh = open(self._stream_path, "a", encoding="utf-8")
            self._fh.write(line + "\n")
            self._fh.flush()
            self._meta["frames"] = seq + 1
        except Exception:
            pass

    def _finish(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
        except Exception:
            pass
        self._flush_meta()

    def _externalize(self, value: Any, default_sr: int) -> Any:
        """递归复制;音频 base64 → @blob/NNNN.wav,视频帧 JPEG → @blob/NNNN.jpg。不改原对象。"""
        if isinstance(value, dict):
            if value.get("source") == "base64" and value.get("format") == "wav" and isinstance(value.get("data"), str):
                # v4 的 system / TTS 参考音频：本身就是 WAV 文件字节，原样落成 wav，不再内联在 session.init 里。
                return {**value, "data": self._stash_wav_file(value["data"])}
            sr = value.get("sample_rate")
            sr = sr if isinstance(sr, int) and sr > 0 else default_sr
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if key in _AUDIO_KEYS and isinstance(item, str):
                    out[key] = self._stash_audio(item, sr)
                elif key == "video_frames" and isinstance(item, list):
                    out[key] = [self._stash_jpeg(x) if isinstance(x, str) else x for x in item]
                else:
                    out[key] = self._externalize(item, default_sr)
            return out
        if isinstance(value, list):
            return [self._externalize(x, default_sr) for x in value]
        return value

    def _next_blob(self, ext: str) -> tuple:
        rel = f"@blob/{self._blob_idx:04d}.{ext}"
        self._blob_idx += 1
        return rel, os.path.join(self._dir, rel[1:])

    def _stash_audio(self, value: str, sample_rate: int) -> str:
        data = _strict_b64decode(value)
        if data is None or len(data) % 4:
            return value                      # 不满足无损条件 → 原样保留
        rel, path = self._next_blob("wav")
        try:
            _write_float32_wav(path, data, sample_rate)
        except Exception:
            return value
        return rel

    def _stash_wav_file(self, value: str) -> str:
        data = _strict_b64decode(value)
        if data is None:
            return value
        rel, path = self._next_blob("wav")
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            return value
        return rel

    def _stash_jpeg(self, value: str) -> str:
        data = _strict_b64decode(value)
        if data is None:
            return value
        rel, path = self._next_blob("jpg")
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            return value
        return rel

    def _flush_meta(self) -> None:
        try:
            tmp = self._meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._meta_path)
        except Exception:
            pass
