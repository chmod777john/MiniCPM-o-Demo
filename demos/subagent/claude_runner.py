"""subagent Demo 的工具执行器：把模型发出的 tool_call 交给 Claude Code（headless ``claude -p``）。

隔离边界（每个 Demo 会话一个工作区）：

- 工作目录 ``workspace/<workspace_id>/``，会话开始时为空目录；
- 权限只放行该目录内的 Read / Edit / Write（Glob、Grep 同受 Read 规则约束）与 WebSearch / WebFetch，
  **Bash 一律禁止**（Bash 能绕开路径规则），``defaultMode=dontAsk``：未放行的调用直接拒绝；
- 用独立的 ``CLAUDE_CONFIG_DIR``，不读用户级 ``~/.claude``（那里放行了 Bash、带全局 CLAUDE.md，
  且会话记录会混进用户的 CloudCLI 列表）。网关地址 / 密钥运行时从用户 settings 的 env 读出，
  以环境变量传入，不写进仓库、不进日志。

回灌形态按 sub_agent 训练数据：

- ``delegate_task``：过程中 Claude 输出的 ``PROGRESS:`` 行 → standalone 进度；最终结论（一段口语）→
  tool result。同一工作区的委派**排队执行并接续同一段 Claude 对话**（训练里追加 / 换题也走它）。
- ``web_search`` / ``calculate``：Claude 按 JSON 回答，这里格式化成训练里的扁平文本
  （``result ok True; result spoken 搜到 N 条…`` / ``result ok True; result result type number; …``）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger("demo_gallery.subagent")

DEMO_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("SUBAGENT_WORKSPACE_ROOT") or (DEMO_DIR / "workspace"))
CLAUDE_CONFIG_DIR = WORKSPACE_ROOT / ".claude-config"
CLAUDE_BIN = os.environ.get("SUBAGENT_CLAUDE_BIN") or "/user/sunweiyue/home/.npm-global/bin/claude"
USER_SETTINGS = Path(os.environ.get("SUBAGENT_USER_SETTINGS") or "/user/sunweiyue/home/.claude/settings.json")
TASK_TIMEOUT_SEC = float(os.environ.get("SUBAGENT_TASK_TIMEOUT_SEC") or 900)
MAX_TURNS = int(os.environ.get("SUBAGENT_MAX_TURNS") or 40)

WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
PROGRESS_RE = re.compile(r"^\s*PROGRESS\s*[:：]\s*(.+?)\s*$")
HANDLED_TOOLS = ("delegate_task", "web_search", "calculate")

EventKind = Literal["progress", "result", "error"]


@dataclass
class TaskEvent:
    """任务事件。progress → 前端以 standalone 注入；result → 以 tool result 回灌。"""

    seq: int
    kind: EventKind
    text: str


@dataclass
class Task:
    """一次工具调用的执行状态。"""

    task_id: str
    workspace_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    language: str
    events: list[TaskEvent] = field(default_factory=list)
    done: bool = False
    started_at: float = field(default_factory=time.time)
    claude_session_id: Optional[str] = None

    def emit(self, kind: EventKind, text: str) -> None:
        self.events.append(TaskEvent(len(self.events), kind, text))
        if kind in ("result", "error"):
            self.done = True


@dataclass
class Workspace:
    """一个 Demo 会话的工作区：独立目录 + 一段可接续的 Claude 对话（供 delegate_task）。"""

    workspace_id: str
    path: Path
    claude_session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claude_session_started: bool = False
    delegate_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: list[str] = field(default_factory=list)


class SubagentRunner:
    """进程内的任务表；Gateway 单进程，足够 Demo 使用。"""

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.workspaces: dict[str, Workspace] = {}

    # ---------------------------------------------------------------- workspace

    def workspace(self, workspace_id: str) -> Workspace:
        """取或建工作区（新建时目录为空）。"""

        if not WORKSPACE_ID_RE.match(workspace_id or ""):
            raise ValueError("workspace_id 只能是 8-64 位字母、数字、下划线或连字符")
        ws = self.workspaces.get(workspace_id)
        if ws is None:
            path = (WORKSPACE_ROOT / workspace_id).resolve()
            path.mkdir(parents=True, exist_ok=True)
            ws = Workspace(workspace_id=workspace_id, path=path)
            self.workspaces[workspace_id] = ws
        return ws

    # ---------------------------------------------------------------- tasks

    def submit(self, *, workspace_id: str, tool_call_id: str, tool_name: str,
               arguments: dict[str, Any], language: str) -> Task:
        """登记并异步启动一次工具执行。"""

        if tool_name not in HANDLED_TOOLS:
            raise ValueError(f"subagent 不处理工具 {tool_name!r}（只处理 {', '.join(HANDLED_TOOLS)}）")
        ws = self.workspace(workspace_id)
        task = Task(task_id=uuid.uuid4().hex[:16], workspace_id=workspace_id, tool_call_id=tool_call_id,
                    tool_name=tool_name, arguments=dict(arguments or {}),
                    language="en" if language == "en" else "zh")
        self.tasks[task.task_id] = task
        ws.tasks.append(task.task_id)
        asyncio.get_running_loop().create_task(self._run(task, ws))
        return task

    def events_after(self, task_id: str, after: int) -> tuple[list[TaskEvent], bool]:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return [e for e in task.events if e.seq >= after], task.done

    async def _run(self, task: Task, ws: Workspace) -> None:
        try:
            if task.tool_name == "delegate_task":
                if not str(task.arguments.get("message") or "").strip():
                    task.emit("result", self._delegate_status(ws, task))
                    return
                async with ws.delegate_lock:      # 同一工作区的委派排队，接续同一段 Claude 对话
                    resume = ws.claude_session_started
                    ws.claude_session_started = True
                    await self._run_claude(task, ws, resume=resume)
            else:
                await self._run_claude(task, ws, resume=False)
        except Exception as exc:  # noqa: BLE001 任何失败都要回给模型，不能让 tool_call 悬着
            logger.exception("[subagent] task %s 失败", task.task_id)
            if not task.done:
                task.emit("error", f"{type(exc).__name__}: {exc}")

    def _delegate_status(self, ws: Workspace, task: Task) -> str:
        """delegate_task 的查询形态（不带 message）：训练数据里没有样例，这里给一句如实的状态。"""

        running = [self.tasks[t] for t in ws.tasks
                   if t != task.task_id and self.tasks[t].tool_name == "delegate_task" and not self.tasks[t].done]
        if not running:
            return "当前没有进行中的后台任务。" if task.language == "zh" else "No background task is running."
        cur = running[-1]
        sec = int(time.time() - cur.started_at)
        n = sum(1 for e in cur.events if e.kind == "progress")
        if task.language == "zh":
            return f"后台任务进行中：已运行 {sec} 秒，已汇报 {n} 条进度。"
        return f"The background task is still running: {sec}s elapsed, {n} progress updates so far."

    # ---------------------------------------------------------------- claude

    async def _run_claude(self, task: Task, ws: Workspace, *, resume: bool) -> None:
        prompt, system = build_prompt(task.tool_name, task.arguments, task.language)
        cmd = [CLAUDE_BIN, "-p", prompt,
               "--output-format", "stream-json", "--verbose",
               "--settings", json.dumps(permission_settings(ws.path)),
               "--setting-sources", "",
               "--max-turns", str(MAX_TURNS),
               "--append-system-prompt", system]
        if task.tool_name == "delegate_task":
            cmd += (["--resume", ws.claude_session_id] if resume else ["--session-id", ws.claude_session_id])
            task.claude_session_id = ws.claude_session_id
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(ws.path), env=claude_env(),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        outcome: dict[str, Any] = {"final_text": None, "is_error": False}

        async def consume() -> None:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("type") == "assistant" and task.tool_name == "delegate_task":
                    for text in _assistant_texts(message):
                        for progress in extract_progress(text):
                            task.emit("progress", progress)
                elif message.get("type") == "result":
                    outcome["final_text"] = str(message.get("result") or "")
                    outcome["is_error"] = bool(message.get("is_error"))
            await proc.wait()

        try:
            await asyncio.wait_for(consume(), timeout=TASK_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            task.emit("error", f"后台执行超时（{int(TASK_TIMEOUT_SEC)} 秒）")
            return
        final_text, is_error = outcome["final_text"], outcome["is_error"]
        if final_text is None or proc.returncode not in (0, None):
            stderr = (await proc.stderr.read()).decode("utf-8", "replace")[-500:] if proc.stderr else ""
            task.emit("error", f"Claude Code 异常退出（exit={proc.returncode}）：{stderr.strip() or '无输出'}")
            return
        if is_error:
            task.emit("error", strip_progress(final_text) or "Claude Code 报告执行失败")
            return
        task.emit("result", format_result(task.tool_name, final_text, task.arguments))


runner = SubagentRunner()


# -------------------------------------------------------------------- 纯函数（有单测）


def permission_settings(workspace: Path) -> dict[str, Any]:
    """只放行工作区内读写与联网；Bash 禁止；未放行即拒绝。"""

    root = f"/{workspace.resolve()}/**"   # Claude Code 权限规则里绝对路径用 // 开头
    return {
        "permissions": {
            "allow": [f"Read({root})", f"Edit({root})", f"Write({root})", "WebSearch", "WebFetch"],
            "deny": ["Bash"],
            "defaultMode": "dontAsk",
        },
        "skipWebFetchPreflight": True,
    }


def claude_env() -> dict[str, str]:
    """子进程环境：去掉父进程里的 Claude / Anthropic 变量，网关配置取自用户 settings 的 env。"""

    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC"))}
    try:
        user_env = json.loads(USER_SETTINGS.read_text(encoding="utf-8")).get("env") or {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"读不到 Claude Code 网关配置 {USER_SETTINGS}: {exc}") from exc
    env.update({k: str(v) for k, v in user_env.items()})
    CLAUDE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    env["CLAUDE_CONFIG_DIR"] = str(CLAUDE_CONFIG_DIR)
    env["HOME"] = str(CLAUDE_CONFIG_DIR)
    return env


def build_prompt(tool_name: str, arguments: dict[str, Any], language: str) -> tuple[str, str]:
    """(用户 prompt, 追加的 system prompt)。"""

    zh = language != "en"
    common = (
        "你是一个语音助手的后台执行者。你只能读写当前工作目录里的文件（用户上传的附件就在这里），"
        "可以用 WebSearch / WebFetch 联网查资料；不能执行命令。"
        if zh else
        "You are the background worker of a voice assistant. You can only read and write files in the "
        "current working directory (uploaded attachments are here) and may use WebSearch / WebFetch; "
        "you cannot run commands."
    )
    if tool_name == "delegate_task":
        message = str(arguments.get("message") or "")
        system = common + (
            "\n执行过程中，每开始一个新步骤之前，先单独输出一行 `PROGRESS: <一句话>`，说明已经查到或做完了什么、"
            "接下来做什么（口语化，不超过 40 字）。最后直接给出最终结论：一段口语化的话，像向助手口头汇报，"
            "不要 markdown、不要列表、不要再写 PROGRESS。"
            if zh else
            "\nBefore starting each new step, output a separate line `PROGRESS: <one sentence>` saying what you "
            "have found or finished and what you will do next (conversational, under 25 words). Finish with the "
            "final conclusion as one conversational paragraph, like an oral report to the assistant: no markdown, "
            "no lists, no more PROGRESS lines."
        )
        return message, system
    if tool_name == "web_search":
        query = str(arguments.get("query") or "")
        limit = int(arguments.get("maxResults") or 5)
        system = common + (
            f"\n用 WebSearch 搜索给定关键词，取最多 {limit} 条结果。最后只输出一个 JSON 对象，不要任何其他文字："
            '{"items":[{"title":"...","url":"...","snippet":"..."}]}'
        )
        return query, system
    if tool_name == "calculate":
        expression = str(arguments.get("expression") or "")
        system = common + (
            "\n精确计算给定的 Python 表达式（Decimal 语义，支持日期时间运算）。最后只输出一个 JSON 对象，不要任何其他文字："
            '{"type":"number","value":<结果>,"approximate":false}；无法计算时输出 {"error":"<原因>"}。'
        )
        return expression, system
    raise ValueError(f"不支持的工具 {tool_name!r}")


def extract_progress(text: str) -> list[str]:
    """从 Claude 的一段文本里取出 PROGRESS 行。"""

    return [m.group(1) for line in text.splitlines() if (m := PROGRESS_RE.match(line))]


def strip_progress(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not PROGRESS_RE.match(line)).strip()


def _assistant_texts(message: dict[str, Any]) -> list[str]:
    content = (message.get("message") or {}).get("content") or []
    return [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"Claude 没有按要求输出 JSON：{text[:200]}")
    return json.loads(match.group(0))


def format_result(tool_name: str, text: str, arguments: dict[str, Any]) -> str:
    """把 Claude 的回答格式化成训练数据里该工具的结果形态。"""

    if tool_name == "delegate_task":
        return strip_progress(text)
    if tool_name == "web_search":
        items = _json_object(text).get("items") or []
        lines = [f"{i}. {it.get('title', '').strip()}（{it.get('url', '').strip()}）{it.get('snippet', '').strip()}"
                 for i, it in enumerate(items, 1)]
        head = f"result ok True; result spoken 搜到 {len(items)} 条（来源 claude_code_websearch，外部网页内容未核实，引用附 URL）："
        return "\n".join([head, *lines])
    if tool_name == "calculate":
        data = _json_object(text)
        if data.get("error"):
            return f"result ok False; result error {data['error']}"
        value = data.get("value")
        kind = data.get("type") or "number"
        approximate = bool(data.get("approximate"))
        return (f"result ok True; result result type {kind}; result result value {value}; "
                f"result result display {value}; result result approximate {approximate}")
    raise ValueError(f"不支持的工具 {tool_name!r}")
