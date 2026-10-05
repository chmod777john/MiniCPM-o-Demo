"""o5_protocol_fsm 逐步走查页面的后端（只用标准库 + SDK）。

    cd /user/sunweiyue/lib/MiniCPM-o-Demo-modelbest
    /user/sunweiyue/runtime/o5_eval/venv-o5-accel/bin/python -m o5_protocol_fsm.explorer.server --port 10017

无状态：前端每次把完整历史发过来，后端从头回放、校验每一步，再把唯一合法的步骤自动填上，
返回下一步的选项。所以页面上看到的每一步都真实经过了一遍 ``O5ProtocolFsm.advance``。
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K

from ..fsm import Abstract, Actor, FsmConfig, O5ProtocolFsm, O5ProtocolViolation, Phase
from ..vocab import O5ProtocolVocab

HERE = Path(__file__).resolve().parent
MAX_AUTOFILL = 64

_COLUMN = {
    Phase.BEFORE_SYSTEM: "system",
    Phase.BETWEEN_UNITS: "prefill",
    Phase.UNIT_OPEN: "prefill",
    Phase.PREFILLED: "spoken",
    Phase.SPOKEN: "spoken",
    Phase.SPOKEN_DONE: "spoken",
    Phase.SPOKEN_CLOSED: "non_spoken",
    Phase.NON_SPOKEN: "non_spoken",
    Phase.NON_SPOKEN_DONE: "non_spoken",
    Phase.NON_SPOKEN_CLOSED: "tail",
}


class Explorer:
    def __init__(self) -> None:
        self.vocab = O5ProtocolVocab.builtin()
        tok = self.vocab.tokenizer
        size = None
        for attr in ("vocab_size", "size"):
            v = getattr(tok, attr, None)
            size = v() if callable(v) else v
            if isinstance(size, int):
                break
        if not isinstance(size, int):
            size = max(self.vocab.id_to_key) + 1
        self.vocab_size = size
        self.n_ordinary = sum(1 for i in range(size) if self.vocab.is_ordinary(i))
        self.n_protocol = len(self.vocab.key_to_id)

    @staticmethod
    def parse_symbol(name: str):
        if name in Abstract.__members__:
            return Abstract[name]
        return K[name]

    def _entry(self, f: O5ProtocolFsm, sym, actor: Actor, auto: bool, n_options: int) -> dict:
        before = f.snapshot()
        col = "end" if sym == Abstract.SESSION_END else _COLUMN.get(f._s.phase, "tail")
        f.advance(sym, actor)
        return {
            "sym": sym.name,
            "label": self.vocab.label(sym),
            "actor": actor.value,
            "auto": auto,
            "col": col,
            "unit": f.snapshot()["unit_index"],
            "n_options": n_options,
            "before": {k: before[k] for k in ("turn_open", "wrapper", "eos_owed")},
        }

    def replay(self, history: list[dict], multi: bool, tts_pad_protocol: str = "revised") -> dict:
        if tts_pad_protocol not in ("revised", "legacy"):
            tts_pad_protocol = "revised"
        f = O5ProtocolFsm(FsmConfig(allow_multi_wrapper_per_slot=multi, tts_pad_protocol=tts_pad_protocol))
        out: list[dict] = []
        error = None
        for i, h in enumerate(history):
            try:
                sym = self.parse_symbol(h["sym"])
                n = len(f.options().all)
                out.append(self._entry(f, sym, Actor(h["actor"]), bool(h.get("auto")), n))
            except (O5ProtocolViolation, KeyError, ValueError) as e:
                error = {"index": i, "code": getattr(e, "code", "bad_request"), "message": str(e)}
                break
        if f._s.phase != Phase.ENDED:
            for _ in range(MAX_AUTOFILL):
                forced = f.options().forced
                if forced is None:
                    break
                out.append(self._entry(f, forced[0], forced[1], True, 1))
        return self._response(f, out, error)

    def _response(self, f: O5ProtocolFsm, history: list[dict], error) -> dict:
        opts = f.options()
        allow_ordinary, special_ids = self.vocab.allowed_ids(opts)
        n_allowed = (self.n_ordinary if allow_ordinary else 0) + len(special_ids)

        def items(syms):
            order = sorted(syms, key=lambda s: (s != Abstract.ORDINARY, s.value))
            return [{"sym": s.name, "label": self.vocab.label(s)} for s in order]

        return {
            "history": history,
            "error": error,
            "state": f.snapshot(),
            "track": opts.track.value,
            "options": {"model": items(opts.model), "framework": items(opts.framework)},
            "can_end": f.can_end_session(),
            "ended": f._s.phase == Phase.ENDED,
            "vocab": {
                "size": self.vocab_size,
                "ordinary": self.n_ordinary,
                "protocol": self.n_protocol,
                "allow_ordinary": allow_ordinary,
                "allowed_special": [self.vocab.label(self.vocab.id_to_key[i]) for i in special_ids],
                "allowed_total": n_allowed,
                "banned_total": self.vocab_size - n_allowed if opts.model else None,
            },
        }


def make_handler(explorer: Explorer):
    index_html = (HERE / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 安静
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, index_html, "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/replay":
                self._send(404, b"not found", "text/plain")
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                resp = explorer.replay(
                    req.get("history") or [],
                    bool(req.get("multi", True)),
                    str(req.get("tts_pad_protocol") or "revised"),
                )
                self._send(200, json.dumps(resp, ensure_ascii=False).encode(), "application/json")
            except Exception as e:  # noqa: BLE001 —— 页面要看到原因，不能只回 500
                self._send(500, json.dumps({"error": {"code": type(e).__name__, "message": str(e)}},
                                           ensure_ascii=False).encode(), "application/json")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10017)
    a = ap.parse_args()
    explorer = Explorer()
    print(f"vocab={explorer.vocab_size} ordinary={explorer.n_ordinary} protocol={explorer.n_protocol}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(explorer)).serve_forever()


if __name__ == "__main__":
    main()
