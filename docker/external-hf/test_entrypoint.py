"""CPU fault injection for container process composition; no model imports."""
import ast
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

source = Path(__file__).with_name('entrypoint.py')
nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)]

def namespace():
    ns = dict(STOPPING=False, CHILDREN=[], ROOT=Path('.'), signal=signal,
              subprocess=subprocess, time=time)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns

class Proc:
    def __init__(self, pid, slow=False):
        self.pid, self.slow, self.waits = pid, slow, 0
    def poll(self):
        return None
    def wait(self, timeout=None):
        self.waits += 1
        if self.slow and self.waits == 1:
            raise subprocess.TimeoutExpired('child', timeout)
        return 0

# Never spawn once shutdown starts.
ns = namespace()
ns['STOPPING'] = True
try:
    ns['launch'](['must-not-spawn'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('spawn after stop')

# Group disappears after poll: remaining children must still be signaled/reaped.
ns = namespace()
procs = [Proc(1, slow=True), Proc(2)]
calls = []
def disappearing_group(pid, sig):
    calls.append((pid, sig))
    if pid == 2 or sig == signal.SIGKILL:
        raise ProcessLookupError()
ns.update(CHILDREN=procs, os=SimpleNamespace(killpg=disappearing_group))
ns['stop']()
assert calls == [(2, signal.SIGTERM), (1, signal.SIGTERM), (1, signal.SIGKILL)]
assert [p.waits for p in procs] == [2, 1]
ns['stop']()  # Idempotent.
assert len(calls) == 3

# Signal arrives within Popen; timeout escalates and child is reaped.
ns = namespace()
proc = Proc(42, slow=True)
calls = []
def popen(*args, **kwargs):
    ns['STOPPING'] = True
    return proc
ns.update(subprocess=SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired),
          os=SimpleNamespace(killpg=lambda pid, sig: calls.append((pid, sig))))
try:
    ns['launch'](['child'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('signal race not handled')
assert calls == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
assert proc.waits == 2
print('ENTRYPOINT_CPU_LIFECYCLE_PASS: prelaunch, disappeared groups, escalation, launch race')
