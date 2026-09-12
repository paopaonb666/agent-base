"""pytest 会话级夹具：测试环境的全局隔离约定。"""

import os

# 默认后端已改为 sqlite（生产语义：对话持久化）。测试必须显式选择自己
# 需要的后端——否则每次 pytest 都会在仓库根创建状态文件。
os.environ.setdefault("CHECKPOINTER_BACKEND", "memory")
