"""文档解析层（M4 前置）：多格式 → 统一的 ``ParsedDocument``。

支持 PDF（pypdf，主依赖）、DOCX（python-docx，``[doc]`` extras，未安装
时该格式不注册并给出可读的安装指引）、TXT / Markdown（直接解码）。
纯函数式：bytes 进、dataclass 出，不做 IO、不碰网络与 LLM，便于单测
与上层复用。

本层刻意不做工具封装：``read_document`` 等工具属于 M4 的工具批次，
在这层之上包装路径安全，超时则由共享池统一添加。限额（输入大小、
输出长度）在本层强制执行，工具层无需重复关心。

公共 API::

    from agent_base.tools.parsing import (
        DocumentParseError,
        ParsedDocument,
        available_formats,
        parse_document,
    )
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePath

from agent_base.tools.parsing import docx
from agent_base.tools.parsing.base import (
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_OUTPUT_CHARS,
    DocumentParseError,
    ParsedDocument,
    parse_page_selection,
)
from agent_base.tools.parsing.pdf import parse_pdf
from agent_base.tools.parsing.text import parse_text

__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DocumentParseError",
    "ParsedDocument",
    "available_formats",
    "parse_document",
    "parse_page_selection",
]

# 扩展名 → 归一化格式名。大小写在查询前归一。
_SUFFIX_FORMATS: dict[str, str] = {
    "pdf": "pdf",
    "docx": "docx",
    "txt": "txt",
    "md": "markdown",
    "markdown": "markdown",
}


def available_formats() -> list[str]:
    """当前环境可用的格式清单；DOCX 依赖 ``[doc]`` extras，可能缺席。"""
    formats = {"pdf", "txt", "markdown"}
    if docx.is_available():
        formats.add("docx")
    return sorted(formats)


def parse_document(
    data: bytes,
    *,
    filename: str | None = None,
    format: str | None = None,
    pages: str | None = None,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ParsedDocument:
    """把文档字节流解析为统一的 ``ParsedDocument``。

    ``filename``  —— 用于按扩展名推断格式（``report.pdf`` / ``notes.md``）
    ``format``    —— 显式格式名，优先于扩展名推断
    ``pages``     —— 页码选择（如 ``"1-3,5"``）；仅 PDF 支持，其他格式忽略
    ``max_input_bytes`` / ``max_output_chars`` —— 限额，默认值见 ``base``；
                     输入超限直接报错，输出超限截断并置 ``truncated`` 标志
    """
    if max_input_bytes <= 0 or max_output_chars <= 0:
        raise DocumentParseError(
            f"解析限额必须为正数：max_input_bytes={max_input_bytes}, "
            f"max_output_chars={max_output_chars}"
        )
    if len(data) > max_input_bytes:
        raise DocumentParseError(
            f"文档超出大小上限：{len(data)} 字节 > 上限 {max_input_bytes} 字节；请拆分或压缩后重试"
        )

    fmt = format if format is not None else _infer_format(filename)
    if fmt == "pdf":
        parsed = parse_pdf(data, pages)
    elif fmt == "docx":
        if not docx.is_available():
            raise DocumentParseError(docx.DOCX_MISSING_MESSAGE)
        parsed = docx.parse_docx(data)
    elif fmt in ("txt", "markdown"):
        parsed = parse_text(data, fmt)
    else:
        raise DocumentParseError(f"不支持的文档格式 {fmt!r}；当前可用：{available_formats()}")

    if len(parsed.text) > max_output_chars:
        parsed = replace(parsed, text=parsed.text[:max_output_chars], truncated=True)
    return parsed


def _infer_format(filename: str | None) -> str:
    """按扩展名推断格式；无法推断时给出"怎么才能推断"的可读错误。"""
    if not filename:
        raise DocumentParseError("无法确定文档格式：请提供带扩展名的 filename，或显式指定 format")
    suffix = PurePath(filename).suffix.lstrip(".").lower()
    fmt = _SUFFIX_FORMATS.get(suffix)
    if fmt is None:
        raise DocumentParseError(
            f"无法识别的文档扩展名 {suffix!r}（filename={filename!r}）；"
            f"当前可用：{available_formats()}"
        )
    return fmt
