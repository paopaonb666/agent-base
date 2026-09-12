"""文档解析层的公共类型、限额与页码选择。

``ParsedDocument`` 是所有解析器的统一返回：纯数据、无 IO、无网络与
LLM 依赖——未来的 ``read_document`` 工具和 RAG 类模块都消费这一个结构。

失败哲学（对 chat-agent 参考实现的显式反转）：chat-agent 的
``_extract_text`` 在解析失败时静默返回空串，上层把"解析不了"当成了
"文档是空的"。本层的契约是坏输入必须抛出带可读信息的
``DocumentParseError``——错误文本会经共享池的 ToolMessage 回给模型，
让它能理解失败原因并自行修正重试。
"""

from __future__ import annotations

from dataclasses import dataclass

# 输入大小封顶默认值（字节）：解析在内存中进行，上限挡住超大文件把
# 工具线程拖入 OOM 的路径。与 core/config.py 的 DOC_PARSE_MAX_INPUT_BYTES
# 默认值保持一致（测试会校验两边相等）。
DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024

# 输出文本封顶默认值（字符）：防止整本 PDF 的文本一次性灌进模型上下文。
# 与 core/config.py 的 DOC_PARSE_MAX_OUTPUT_CHARS 默认值保持一致。
DEFAULT_MAX_OUTPUT_CHARS = 50_000


class DocumentParseError(ValueError):
    """解析失败（格式不支持、文件损坏、超出限额、依赖缺失等）。"""


@dataclass(frozen=True)
class ParsedDocument:
    """统一解析结果。

    ``text``       —— 提取出的纯文本（可能被截断，见 ``truncated``）
    ``format``     —— 归一化的格式名（pdf / docx / txt / markdown）
    ``pages``      —— 文档总页数（仅 PDF 设置；供上层提示"还有更多页"）
    ``paragraphs`` —— 段落数（docx 为正文段落数；文本格式为非空行数）
    ``truncated``  —— 输出超过 ``max_output_chars`` 被截断时为 True
    """

    text: str
    format: str
    pages: int | None = None
    paragraphs: int | None = None
    truncated: bool = False


def parse_page_selection(spec: str, total: int) -> list[int]:
    """把 ``"1-3,5"`` 这样的页码选择解析成去重升序的 1-based 页号列表。

    页码通常来自 LLM 的工具参数，因此任何畸形或越界输入都抛出带可读
    信息的 ``DocumentParseError``——报错文本经 ToolMessage 回给模型后，
    它能理解规则并自行修正重试。
    """
    selected: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            raise DocumentParseError(f"无效的页码选择 {spec!r}：存在空的页码项")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2:
                raise DocumentParseError(f"无效的页码选择 {spec!r}：{token!r} 不是页码或页码区间")
            start, end = _parse_page_number(bounds[0], spec), _parse_page_number(bounds[1], spec)
            if end < start:
                raise DocumentParseError(
                    f"无效的页码选择 {spec!r}：区间终点 {end} 小于起点 {start}"
                )
        else:
            start = end = _parse_page_number(token, spec)
        if start < 1:
            raise DocumentParseError(f"无效的页码选择 {spec!r}：页码从 1 开始，收到了 {start}")
        if end > total:
            raise DocumentParseError(
                f"页码超出范围：文档共 {total} 页，但页码选择 {spec!r} 请求到了第 {end} 页"
            )
        selected.update(range(start, end + 1))
    return sorted(selected)


def _parse_page_number(token: str, spec: str) -> int:
    """把单个页码 token 解析成整数；畸形输入给出指向原始 spec 的报错。"""
    try:
        return int(token)
    except ValueError as exc:
        raise DocumentParseError(f"无效的页码选择 {spec!r}：{token!r} 不是整数页码") from exc
