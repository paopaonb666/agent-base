"""基座存储的数据模型。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThreadIndex:
    """一行会话索引（S1 线程作用域的数据模型基础）。

    ``thread_id`` 是带模块前缀的完整 id（``module:user_thread_id``）；
    属主（user_id）在 invoke 时写入，线程端点的列表/属主校验据此执行。
    """

    thread_id: str
    user_id: str
    module: str = ""
    title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
