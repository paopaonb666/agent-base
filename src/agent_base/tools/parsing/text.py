"""TXT / Markdown 解析器：直接解码（UTF-8 优先，容错 GBK 系）。

两种格式共用同一实现——解析层只负责忠实地取出文本；按标题切分等
"理解"逻辑留给上层工具与未来的 RAG 模块。编码策略：UTF-8（带 BOM
剥离）严格解码优先，失败时退回 GB18030（GBK/GB2312 的官方超集）；
两者都失败说明输入不是文本或是不认识的编码，按契约抛出可读错误，
绝不静默替换字符。
"""

from __future__ import annotations

from agent_base.tools.parsing.base import DocumentParseError, ParsedDocument


def decode_text(data: bytes) -> str:
    """把字节流解码为文本：UTF-8 严格解码优先，失败退回 GB18030。"""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("gb18030")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(
            "文本解码失败：输入既不是有效的 UTF-8，也不是 GBK/GB18030 编码"
        ) from exc


def parse_text(data: bytes, fmt: str) -> ParsedDocument:
    """解析 TXT / Markdown 字节流；``fmt`` 是归一化后的格式名。"""
    text = decode_text(data)
    paragraphs = sum(1 for line in text.splitlines() if line.strip())
    return ParsedDocument(text=text, format=fmt, paragraphs=paragraphs)
