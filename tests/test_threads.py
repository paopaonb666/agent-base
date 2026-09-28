"""ThreadId 值对象（P1-5）的测试：round-trip + 非法输入拒绝。"""

from __future__ import annotations

import pytest

from agent_base.core.threads import SEPARATOR, ThreadId, ThreadIdError


def test_round_trip() -> None:
    tid = ThreadId(module="chat", raw="abc-123")
    assert str(tid) == "chat:abc-123"
    parsed = ThreadId.parse("chat:abc-123")
    assert parsed == tid


def test_parse_splits_on_first_separator_only() -> None:
    # module 侧不允许分隔符，raw 侧也不允许——解析端第一个冒号即分界，
    # 其余冒号属于 raw，会在构造校验里被拒绝（而不是静默拆错）。
    with pytest.raises(ThreadIdError):
        ThreadId.parse("chat:weird:id")


def test_invalid_constructions_rejected() -> None:
    with pytest.raises(ThreadIdError):
        ThreadId(module="", raw="x")
    with pytest.raises(ThreadIdError):
        ThreadId(module="chat", raw="")
    with pytest.raises(ThreadIdError):
        ThreadId(module="chat", raw=f"a{SEPARATOR}b")
    with pytest.raises(ThreadIdError):
        ThreadId(module=f"ch{SEPARATOR}at", raw="x")


def test_parse_rejects_non_namespaced() -> None:
    with pytest.raises(ThreadIdError):
        ThreadId.parse("no-separator")
    with pytest.raises(ThreadIdError):
        ThreadId.parse("")
    with pytest.raises(ThreadIdError):
        ThreadId.parse(":raw-only")
    with pytest.raises(ThreadIdError):
        ThreadId.parse("module:")


def test_try_parse_returns_none_instead_of_raising() -> None:
    # 直调工具 / CLI ad-hoc 场景没有命名空间：宽松解析给 None，
    # 调用方据此回退自己的默认值（"" 或 "*"）。
    assert ThreadId.try_parse("") is None
    assert ThreadId.try_parse("bare") is None
    assert ThreadId.try_parse("chat:t") == ThreadId(module="chat", raw="t")
