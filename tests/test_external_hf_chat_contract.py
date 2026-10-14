"""CPU checks for desktop chat option routing into external HF, without model load."""
from __future__ import annotations
import ast
import base64
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from py_backend.chat_util import parse_worker_chat_request_message, ChatRequestError

source = ast.parse((ROOT / "core/processors/unified.py").read_text())
view = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == "ChatView")
ns = dict(MiniCPMOProcessorMixin=object, np=np, torch=torch, time=time,
          logger=logging.getLogger(__name__))
exec(compile(ast.Module(body=[view], type_ignores=[]), "unified.py", "exec"), ns)
ChatView = ns["ChatView"]

class Model:
    _external_hf_chat = True
    def __init__(self): self.calls=[]; self.closed=False
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        try:
            if kwargs["generate_audio"]:
                yield torch.tensor([.25, -.25]), "你好"
            else:
                yield "你好"
        finally:
            self.closed=True

for audio in (False, True):
    m=Model(); v=ChatView(m)
    history=[{"role":"user", "content":"A"}, {"role":"assistant", "content":"B"},
             {"role":"user", "content":"C"}]
    assert v.prefill("one", history, omni_mode=True, max_slice_nums=2,
                     use_tts_template=audio, enable_thinking=False)==""
    assert not m.calls
    chunks=list(v.streaming_generate("one", generate_audio=audio, do_sample=False,
                                    max_new_tokens=12, length_penalty=1.7))
    assert m.closed and len(m.calls)==1
    call=m.calls[0]
    assert call["msgs"] is history and call["stream"] is True
    assert call["do_sample"] is False and call["max_new_tokens"]==12
    assert call["max_slice_nums"]==2 and call["length_penalty"]==1.7
    assert call["omni_mode"] is True and call["enable_thinking"] is False
    assert chunks[0].text_delta=="你好"
    if audio:
        assert np.array_equal(np.frombuffer(base64.b64decode(chunks[0].audio_data),dtype=np.float32), [.25,-.25])
    try:
        list(v.streaming_generate("one"))
    except ValueError: pass
    else: raise AssertionError("stale prepared input was reused")

q=parse_worker_chat_request_message({"type":"chat.request", "payload":{
    "messages":[{"role":"user", "content":"hello"}], "streaming":False,
    "generation":{"do_sample":False,"max_new_tokens":7,"length_penalty":1.4},
    "image":{"max_slice_nums":2}, "tts":{"enabled":True}, "omni_mode":True}})
assert not q.do_sample and not q.streaming and q.max_new_tokens==7
assert q.length_penalty==1.4 and q.max_slice_nums==2 and q.generate_audio
assert q.omni_mode and q.use_tts_template
for payload in ({}, {"messages":[{}],"generation":{"max_new_tokens":0}},
                {"messages":[{}],"tts":{"enabled":True},"enable_thinking":True}):
    try: parse_worker_chat_request_message({"type":"chat.request","payload":payload})
    except ChatRequestError: pass
    else: raise AssertionError("invalid chat request accepted")
print("HF chat bridge: complete history, options, text/audio conversion, one-shot lifecycle PASS")

class CompleteModel:
    _external_hf_chat = True
    def __init__(self): self.calls=[]
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return "nonstream result"
m=CompleteModel();v=ChatView(m)
history=[{"role":"user","content":"完整历史"}]
v.prefill("complete",history,use_tts_template=False)
assert v.generate("complete",generate_audio=False,do_sample=False,max_new_tokens=7)=="nonstream result"
assert len(m.calls)==1 and m.calls[0]["msgs"] is history
assert m.calls[0]["stream"] is False and m.calls[0]["do_sample"] is False
assert v._chat_inputs is None
print("HF nonstream view bridge and invalid speech/thinking request rejection PASS")
