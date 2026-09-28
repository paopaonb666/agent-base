"""线程标识值对象（P1-5）：把 ``{module}:{user_thread_id}`` 的隐式字符串
协议显式化。

一处定义、处处引用：core 的工具执行上下文、memory 的作用域解析与
server 的线程路由都经 ``ThreadId`` 拼接/解析，分隔符约定不再散落在
各文件的 ``split`` / f-string 里。raw 部分含分隔符在构造时即被拒绝
（静默错位是这类字符串协议最隐蔽的故障形态——module 拆错、作用域
串号都不会报错）。
"""

from __future__ import annotations

from dataclasses import dataclass

SEPARATOR = ":"


class ThreadIdError(ValueError):
    """非法的线程标识（空组成部分 / 含分隔符 / 解析输入缺分隔符）。"""


@dataclass(frozen=True)
class ThreadId:
    """命名空间线程标识：module（图/模块名）+ raw（用户侧 thread id）。

    ``str()`` 产出存储与 LangGraph configurable 使用的
    ``{module}{SEPARATOR}{raw}`` 形态。
    """

    module: str
    raw: str

    def __post_init__(self) -> None:
        if not self.module:
            raise ThreadIdError("ThreadId.module 不能为空")
        if not self.raw:
            raise ThreadIdError("ThreadId.raw 不能为空")
        if SEPARATOR in self.module:
            raise ThreadIdError(f"ThreadId.module 不能包含分隔符 {SEPARATOR!r}: {self.module!r}")
        if SEPARATOR in self.raw:
            raise ThreadIdError(f"ThreadId.raw 不能包含分隔符 {SEPARATOR!r}: {self.raw!r}")

    def __str__(self) -> str:
        return f"{self.module}{SEPARATOR}{self.raw}"

    @classmethod
    def parse(cls, value: str) -> ThreadId:
        """从存储/checkpointer 里的完整字符串解析；格式非法抛 ThreadIdError。"""
        module, sep, raw = value.partition(SEPARATOR)
        if not sep or not module or not raw:
            raise ThreadIdError(f"不是命名空间线程标识（应为 module{SEPARATOR}raw）: {value!r}")
        return cls(module=module, raw=raw)

    @classmethod
    def try_parse(cls, value: str) -> ThreadId | None:
        """宽松解析：空串 / 无分隔符 / 格式非法返回 None（调用方自行给默认值）。"""
        if not value:
            return None
        try:
            return cls.parse(value)
        except ThreadIdError:
            return None
