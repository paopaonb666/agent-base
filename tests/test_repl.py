"""python_repl 的测试（M5）：执行、错误回传、超时终止、输出封顶。

POSIX 的 rlimit 路径由 CI 的 Linux job 覆盖（skipif 反向跳过）。
"""

from __future__ import annotations

import sys

import pytest

from agent_base.tools import repl
from agent_base.tools.repl import python_repl


def test_repl_runs_code_and_returns_stdout() -> None:
    rendered = python_repl.invoke({"code": "print(1 + 1)"})
    assert "exit code: 0" in rendered
    assert "--- stdout ---" in rendered
    assert "2" in rendered


def test_repl_reports_stderr() -> None:
    rendered = python_repl.invoke({"code": "import sys; print('warn', file=sys.stderr)"})
    assert "exit code: 0" in rendered
    assert "--- stderr ---" in rendered
    assert "warn" in rendered


def test_repl_exception_reaches_stderr() -> None:
    rendered = python_repl.invoke({"code": "raise ValueError('boom')"})
    assert "exit code: 1" in rendered
    assert "ValueError" in rendered and "boom" in rendered


def test_repl_rejects_empty_code() -> None:
    with pytest.raises(ValueError, match="code 不能为空"):
        python_repl.invoke({"code": "   "})


def test_repl_timeout_terminates_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repl, "REPL_SUBPROCESS_TIMEOUT_SECONDS", 0.5)
    rendered = python_repl.invoke({"code": "import time; time.sleep(5)"})
    assert "执行超时" in rendered
    assert "0 秒" in rendered


def test_repl_truncates_huge_output() -> None:
    rendered = python_repl.invoke({"code": "print('x' * 50_000)"})
    assert "已截断" in rendered
    assert len(rendered) < 50_000


@pytest.mark.skipif(sys.platform == "win32", reason="rlimit 仅 POSIX 平台生效")
def test_repl_applies_posix_limits() -> None:
    # 在 Linux CI 上真实执行 preexec 限额路径：正常短代码不受限额影响。
    rendered = python_repl.invoke({"code": "print('limits-ok')"})
    assert "limits-ok" in rendered
