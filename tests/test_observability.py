"""阶段 2 可观测性的测试：request_id 传播 + 结构化日志。

覆盖 chat-agent 评审中的 A1 继承：request id 必须从 contextvar 流入每一条
日志记录（filter），且格式化必须同时支持人类可读（开发）和 JSON（生产）
输出。追踪仍由环境开关控制。
"""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from agent_base.extensions.observability import (
    HumanFormatter,
    JsonFormatter,
    RequestIdFilter,
    get_request_id,
    log_tracing_config,
    new_request_id,
    request_id,
    setup_logging,
)


class TestRequestIdContext:
    def test_empty_outside_scope(self) -> None:
        assert get_request_id() == ""

    def test_scope_sets_and_restores(self) -> None:
        assert get_request_id() == ""
        with request_id("rid-1"):
            assert get_request_id() == "rid-1"
        assert get_request_id() == ""

    def test_scope_generates_when_absent(self) -> None:
        with request_id() as rid:
            assert rid
            assert get_request_id() == rid

    def test_generated_ids_are_unique(self) -> None:
        assert new_request_id() != new_request_id()

    def test_nested_scope_restores_outer(self) -> None:
        with request_id("outer"):
            with request_id("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"


class TestRequestIdFilter:
    def test_injects_request_id(self) -> None:
        f = RequestIdFilter()
        record = logging.LogRecord("m", logging.INFO, "f", 1, "msg", None, None)
        with request_id("rid-abc"):
            assert f.filter(record) is True
            assert record.request_id == "rid-abc"

    def test_injects_dash_when_unset(self) -> None:
        f = RequestIdFilter()
        record = logging.LogRecord("m", logging.INFO, "f", 1, "msg", None, None)
        f.filter(record)
        assert record.request_id == "-"


class TestFormatters:
    def test_human_formatter_includes_request_id(self) -> None:
        f = HumanFormatter()
        record = logging.LogRecord("m", logging.INFO, "f", 1, "hello", None, None)
        record.request_id = "rid-xyz"  # 真实流程中由 filter 注入
        line = f.format(record)
        assert "rid-xyz" in line
        assert "INFO" in line

    def test_json_formatter_structure(self) -> None:
        f = JsonFormatter()
        record = logging.LogRecord("m", logging.INFO, "f", 1, "hello", None, None)
        record.request_id = "rid-json"
        payload = json.loads(f.format(record))
        assert payload["level"] == "INFO"
        assert payload["request_id"] == "rid-json"
        assert payload["message"] == "hello"
        assert payload["logger"] == "m"

    def test_json_formatter_escapes_exc_info(self) -> None:
        f = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            record = logging.LogRecord(
                "m",
                logging.ERROR,
                "f",
                1,
                "failed",
                None,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            record.request_id = "rid-e"
            payload = json.loads(f.format(record))
            assert "ValueError" in payload["exc_info"]


class TestSetupLogging:
    def test_setup_is_idempotent(self) -> None:
        root = logging.getLogger()
        # 其他测试（CLI）也会调用 setup_logging；本测试不应依赖运行顺序，
        # 因此从干净的状态开始。
        for handler in list(root.handlers):
            if handler.get_name() == "agent_base.console":
                root.removeHandler(handler)
        before = len(root.handlers)
        setup_logging()
        after_first = len(root.handlers)
        setup_logging()
        assert len(root.handlers) == after_first
        assert after_first == before + 1  # 恰好新增了一个具名处理器

    def test_setup_replaces_formatter_for_json(self) -> None:
        setup_logging(json_lines=True)
        handler = next(
            h for h in logging.getLogger().handlers if h.get_name() == "agent_base.console"
        )
        assert isinstance(handler.formatter, JsonFormatter)

    def test_any_logger_gets_request_id(self) -> None:
        """端到端：一个基座从不拥有的 logger 依然携带 request_id。

        针对 root-logger-filter 缺陷的回归测试：``Logger.filter`` 只会查询
        logger 自己的过滤器，所以为了让任意 logger（如 langchain）发出的
        记录也能被增强，过滤器必须放在处理器上。处理器的流被替换为内存
        缓冲区，以保持独立于 pytest 的 stderr 捕获。
        """
        setup_logging()
        handler = next(
            h for h in logging.getLogger().handlers if h.get_name() == "agent_base.console"
        )
        buf = io.StringIO()
        handler.stream = buf
        try:
            with request_id("rid-e2e"):
                logging.getLogger("some.third.party").error("boom")
            assert "rid-e2e" in buf.getvalue()
        finally:
            handler.stream = sys.stderr


class TestTracingConfig:
    def test_reports_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        # 必须不抛异常；输出通过下面的 caplog 断言。
        log_tracing_config()

    def test_reports_langsmith_on(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        with caplog.at_level(logging.INFO, logger="agent_base.tracing"):
            log_tracing_config()
        assert "langsmith=on" in caplog.text

    def test_reports_langfuse_on(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        with caplog.at_level(logging.INFO, logger="agent_base.tracing"):
            log_tracing_config()
        assert "langfuse=on" in caplog.text


def test_env_flag_parsing() -> None:
    from agent_base.extensions.observability import _env_flag

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert _env_flag(truthy) is True
    for falsy in ("", "0", "false", "no", "off", None):
        assert _env_flag(falsy) is False
