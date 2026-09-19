"""HTTP 层的序列化与内容嗅探（H1 拆分）：消息 ↔ 前端结构、图片类型。"""

from __future__ import annotations

from typing import Any

# 图片附件（多模态对话）：按 magic bytes 嗅探真实类型——扩展名可伪装，
# 内容头不会。webp 的 RIFF/WEBP 头单独检查。
_IMAGE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpeg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
)
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
# filestore 中图片附件的归一化 format 值（mime 去前缀）。
IMAGE_STORED_FORMATS = {"png", "jpeg", "webp", "gif"}


def _sniff_image_mime(data: bytes) -> str | None:
    """按 magic bytes 识别图片真实类型；无法识别返回 None。"""
    for signature, mime, _ in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _extract_text(content: Any) -> str:
    """提取消息 content 里的纯文本：str 直接返回；列表型 content
    （部分 provider 的多段内容）拼接各段的 text 字段，避免把
    Python repr 原样发给客户端。"""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _serialize_message(message: Any) -> dict[str, Any] | None:
    """把一条 LangChain 消息序列化为前端可渲染的简单结构。

    返回 ``None`` 表示这条消息不该显示（例如仅含 tool_calls 的 AI 消息）。
    前端 agent-base-ui 用它与 invoke 流式事件对齐，以便回放历史会话。

    工具调用可观测（M5）：AI 消息携带 ``tool_calls``（id/name/args——
    模型给定的调用参数），tool 消息携带 ``tool_call_id``/``status``——
    前端按 id 把两者配对，回放出完整的"参数 + 结果"调用面板。仅含
    tool_calls 的 AI 消息也保留（它是调用参数的唯一持久化位置）。
    """
    mtype = getattr(message, "type", "")
    text = _extract_text(getattr(message, "content", ""))
    if mtype == "system":
        # 附件注入的系统消息：内容只在当轮给模型，历史回放由 human 消息
        # 上的附件元数据承载——blob 永不上屏。
        return None
    if mtype == "human":
        attachments = (getattr(message, "additional_kwargs", {}) or {}).get("attachments")
        out: dict[str, Any] = {"role": "human", "content": text}
        if isinstance(attachments, list) and attachments:
            out["attachments"] = attachments
        return out
    if mtype == "ai":
        tool_calls: list[dict[str, Any]] = [
            {
                "id": str(tc.get("id") or ""),
                "name": str(tc.get("name") or ""),
                "args": tc.get("args") if isinstance(tc.get("args"), dict) else {},
            }
            for tc in (getattr(message, "tool_calls", None) or [])
            if isinstance(tc, dict)
        ]
        if not text and not tool_calls:
            return None  # 既无文本也无调用的空 AI 消息，没有可展示内容
        return {
            "role": "assistant",
            "content": text,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }
    if mtype == "tool":
        raw_status = str(getattr(message, "status", "") or "success")
        return {
            "role": "tool",
            "name": str(getattr(message, "name", "") or ""),
            "content": text,
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            "status": "error" if raw_status == "error" else "ok",
        }
    return None
