"""CPU check of actual API coroutine: cancelled text drains, speech closes."""
import ast
import asyncio
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, Optional

source = ast.parse(Path('py_backend/server.py').read_text())
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'BackendProtocolSession')
method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_stream_turn_based')
ns = dict(asyncio=asyncio, Any=Any, Optional=Optional)
exec(compile(ast.Module(body=[method], type_ignores=[]), '<real _stream_turn_based>', 'exec'), ns)

async def case(audio):
    consumed=[]; closed=[]; cancel=Event()
    class Backend:
        def chat_streaming_generate(self, **kwargs):
            try:
                for i in range(20):
                    time.sleep(.005)
                    consumed.append(i)
                    yield SimpleNamespace(text_delta=str(i), audio_data=None)
            finally: closed.append(True)
    session=SimpleNamespace(backend=Backend(),session_id='test',_chat_cancel=cancel,closed=False)
    async def send(*args,**kwargs):
        session.closed=True;cancel.set()
    session.send_output_delta=send
    session.send=send
    session._safe_metrics=lambda:{}
    request=SimpleNamespace(generate_audio=audio,max_new_tokens=20,do_sample=False)
    await ns['_stream_turn_based'](session,request,response_id='r',input_id='i')
    assert closed
    if audio: assert len(consumed)<20,consumed
    else: assert consumed==list(range(20)),consumed
    return len(consumed)
async def main():
    print({'text_consumed_after_cancel':await case(False),'audio_consumed_after_cancel':await case(True)})
asyncio.run(main())
