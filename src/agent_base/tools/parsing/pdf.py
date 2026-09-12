"""PDF 解析器（pypdf）。

pypdf 纯 Python 且轻量，作为主依赖随基座安装——PDF 是文档读取的主
格式，应当开箱即用。逐页 ``extract_text`` 后按页拼接（沿用 chat-agent
已验证的做法）；``pages`` 参数支持 ``"1-3,5"`` 这类页码选择，让模型
可以分批读取大文档而不是一次性吞下全文。

扫描版 PDF（纯图片页）的提取结果是空文本——这不是错误：返回空文本
并保留页数元信息，由上层工具决定如何向模型解释（例如提示"可能是
扫描件"）。
"""

from __future__ import annotations

import io

from pypdf import PdfReader

from agent_base.tools.parsing.base import (
    DocumentParseError,
    ParsedDocument,
    parse_page_selection,
)


def parse_pdf(data: bytes, pages: str | None = None) -> ParsedDocument:
    """解析 PDF 字节流；``pages`` 为可选页码选择（如 ``"1-3,5"``，1-based）。"""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentParseError("PDF 已加密，无法解析：请先提供解密后的文件")
        total = len(reader.pages)
        page_objs = reader.pages
        if pages is not None:
            selected = parse_page_selection(pages, total)
            page_objs = [reader.pages[number - 1] for number in selected]
        texts = [page.extract_text() or "" for page in page_objs]
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"PDF 解析失败：文件可能已损坏或不是有效的 PDF（{type(exc).__name__}: {exc}）"
        ) from exc
    return ParsedDocument(text="\n".join(texts), format="pdf", pages=total)
