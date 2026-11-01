"""pytest 会话级夹具：测试环境的全局隔离约定。"""

import os

# 默认后端已改为 sqlite（生产语义：对话持久化）。测试必须显式选择自己
# 需要的后端——否则每次 pytest 都会在仓库根创建状态文件。
os.environ.setdefault("CHECKPOINTER_BACKEND", "memory")

# Redis 共享状态服务默认开启（ADR-012），但核心套件不依赖它：默认关闭，
# 全部能力回退进程内实现；integration 标记的 Redis 测试自己显式开启并
# 连接（CI 的 test job 挂了 redis 服务容器）。
os.environ.setdefault("REDIS_ENABLED", "false")
