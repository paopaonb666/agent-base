"""Light tests for the CLI entrypoint (no LLM/network)."""

from __future__ import annotations

import pytest

from agent_base import __version__
from agent_base.entrypoints.cli import build_parser, main


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
