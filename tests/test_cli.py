"""CLI 入口的轻量测试（无 LLM / 网络）。"""

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from agent_base import __version__
from agent_base.core.bootstrap import UnknownModuleError
from agent_base.entrypoints import cli
from agent_base.entrypoints.cli import build_parser, main


class _FakeGraph:
    """最小化的图替身：追加一条脚本化的 assistant 回复。"""

    def __init__(self, reply: str = "hello") -> None:
        self._reply = reply
        self.config: dict[str, Any] | None = None

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.config = config
        user_text = str(payload["messages"][-1].content)
        content = f"{self._reply} to: {user_text}" if self._reply else ""
        return {"messages": [*payload["messages"], AIMessage(content=content)]}


class _FakeRuntime:
    """最小化的运行时替身，覆盖 CLI 的各个触点。"""

    settings = SimpleNamespace(
        observability=SimpleNamespace(log_json=False), agent_recursion_limit=25
    )
    closed = False

    def __init__(self, graph: _FakeGraph | None = None) -> None:
        self._graph = graph or _FakeGraph()

    def graph(self, module_name: str) -> _FakeGraph:
        if module_name == "missing_xyz":
            raise UnknownModuleError(
                f"module {module_name!r} is not enabled; available: chat, writer, supervisor"
            )
        return self._graph

    async def close(self) -> None:
        self.closed = True


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime: _FakeRuntime | None = None
) -> _FakeRuntime:
    runtime = runtime or _FakeRuntime()

    async def _create_runtime(**kwargs: Any) -> _FakeRuntime:
        return runtime

    monkeypatch.setattr(cli, "create_runtime", _create_runtime)
    return runtime


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.module == "chat"
    assert args.message is None


def test_parser_message_flag() -> None:
    args = build_parser().parse_args(["--module", "chat", "--message", "hi"])
    assert args.module == "chat"
    assert args.message == "hi"


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_python_m_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m agent_base` 会运行 main()（模块 __main__ 垫片）。"""
    monkeypatch.setattr(sys.modules["agent_base.entrypoints.cli"], "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("agent_base.__main__", run_name="__main__")
    assert exc.value.code == 0


async def test_invoke_namespaces_thread_per_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """thread id 必须按模块划分命名空间（共用的 checkpointer）。"""
    runtime = _FakeRuntime()
    graph = runtime._graph
    messages = await cli._invoke(runtime, "chat", "thread-1", "hi")
    assert graph.config == {
        "recursion_limit": 25,
        "configurable": {"thread_id": "chat:thread-1"},
    }
    assert [m.content for m in messages] == ["hi", "hello to: hi"]


async def test_one_shot_prints_reply_and_thread_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _install_runtime(monkeypatch)
    await cli._one_shot(runtime, "chat", "t1", "hi")
    out, err = capsys.readouterr()
    assert "hello to: hi" in out
    assert "thread_id: t1" in err


async def test_one_shot_without_reply_prints_thread_id_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """空的 assistant 回复绝不能打印空行。"""
    runtime = _install_runtime(monkeypatch, _FakeRuntime(_FakeGraph(reply="")))
    await cli._one_shot(runtime, "chat", "t1", "hi")
    out, err = capsys.readouterr()
    assert out == ""
    assert "thread_id: t1" in err


async def test_interactive_turn_and_quit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["hi", "", "quit"])  # 空行被跳过，quit 退出

    def _input(prompt: str = "") -> str:
        print(prompt, end="")
        return next(answers)

    monkeypatch.setattr("builtins.input", _input)
    runtime = _install_runtime(monkeypatch)
    await cli._interactive(runtime, "chat", "t1")
    out, _ = capsys.readouterr()
    assert "agent> hello to: hi" in out
    assert "thread_id: t1" in out


async def test_interactive_eof_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    def _eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    runtime = _install_runtime(monkeypatch)
    await cli._interactive(runtime, "chat", "t1")  # 必须正常返回，而非抛异常


async def test_run_one_shot_returns_0_and_closes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _install_runtime(monkeypatch)
    args = SimpleNamespace(module="chat", message="hi", thread_id="abc123")
    assert await cli._run(args) == 0
    assert runtime.closed
    out, err = capsys.readouterr()
    assert "hello to: hi" in out
    assert "thread_id: abc123" in err


async def test_run_settings_error_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _fail(**kwargs: Any) -> _FakeRuntime:
        raise cli.SettingsError("bad config")

    monkeypatch.setattr(cli, "create_runtime", _fail)
    args = SimpleNamespace(module="chat", message=None, thread_id=None)
    assert await cli._run(args) == 2
    assert "configuration error: bad config" in capsys.readouterr().err


async def test_run_unknown_module_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_runtime(monkeypatch)
    args = SimpleNamespace(module="missing_xyz", message="hi", thread_id=None)
    assert await cli._run(args) == 2
    assert "error:" in capsys.readouterr().err


async def test_unknown_module_still_closes_runtime(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """错误退出路径也必须释放 checkpointer 连接。"""
    runtime = _install_runtime(monkeypatch)
    args = SimpleNamespace(module="missing_xyz", message="hi", thread_id=None)
    assert await cli._run(args) == 2
    assert runtime.closed


async def test_internal_key_error_propagates_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图内部真正的 KeyError 不许被吞成"模块不存在"文案。"""

    class _ExplodingGraph(_FakeGraph):
        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            raise KeyError("internal state key")

    runtime = _install_runtime(monkeypatch, _FakeRuntime(_ExplodingGraph()))
    args = SimpleNamespace(module="chat", message="hi", thread_id=None)
    with pytest.raises(KeyError, match="internal state key"):
        await cli._run(args)
    assert runtime.closed


def test_main_one_shot_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_runtime(monkeypatch)
    assert main(["--message", "hi"]) == 0


def test_main_unhandled_error_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _explode(**kwargs: Any) -> _FakeRuntime:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "create_runtime", _explode)
    assert main(["--message", "hi"]) == 1
    assert "error: boom" in capsys.readouterr().err
