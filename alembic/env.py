"""Alembic 迁移环境：从项目配置动态构造 MySQL DSN。

本环境只负责"连接到哪个库"，不声明任何模型元数据——因为项目当前
没有 SQLAlchemy Model，迁移以纯 SQL 形式编写（见 versions/）。
连接参数优先取环境变量（供 CI 覆盖），否则回退到
``agent_base.core.config.Settings``（会读取 .env）。
"""

import os
from logging.config import fileConfig
from urllib.parse import quote

from alembic import context
from sqlalchemy import engine_from_config, pool

# 这是 Alembic 的 Config 对象，提供对 .ini 文件中值的访问。
config = context.config

# 解释配置文件以设置 Python 日志。
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 项目目前使用 langgraph-checkpoint-mysql（纯 SQL），没有 ORM 模型元数据，
# 因此 autogenerate 不可用；迁移以手写 SQL 为主。
target_metadata = None


def _mysql_dsn() -> str:
    """组装 MySQL DSN，供 online / offline 两种模式使用。

    优先级：显式 ``AGENT_BASE_DB_*`` 环境变量 > ``.env`` 中的
    ``CHECKPOINTER_MYSQL_*``。这样 CI / 外部环境可以在不落地 .env 的情况下
    覆盖连接参数。
    """
    try:
        from agent_base.core.config import Settings

        s = Settings()
        host = os.environ.get("AGENT_BASE_DB_HOST", s.checkpointer_mysql_host)
        port = os.environ.get("AGENT_BASE_DB_PORT", str(s.checkpointer_mysql_port))
        user = os.environ.get("AGENT_BASE_DB_USER", s.checkpointer_mysql_user)
        password = os.environ.get(
            "AGENT_BASE_DB_PASSWORD", s.checkpointer_mysql_password.get_secret_value()
        )
        db = os.environ.get("AGENT_BASE_DB_NAME", s.checkpointer_mysql_database)
    except Exception:
        # 兜底：仅有环境变量，避免在无项目配置时崩溃。
        host = os.environ.get("AGENT_BASE_DB_HOST", "127.0.0.1")
        port = os.environ.get("AGENT_BASE_DB_PORT", "3306")
        user = os.environ.get("AGENT_BASE_DB_USER", "root")
        password = os.environ.get("AGENT_BASE_DB_PASSWORD", "")
        db = os.environ.get("AGENT_BASE_DB_NAME", "agent_base")
    return (
        f"mysql+pymysql://{quote(user)}:{quote(password)}@{host}:{int(port)}/{quote(db)}"
        "?charset=utf8mb4"
    )


def run_migrations_offline() -> None:
    """以 'offline' 模式运行迁移。

    只配置 URL 而不创建 Engine，因此甚至不需要 DBAPI 可用。调用
    context.execute() 会把给定字符串输出为 SQL，可用于生成迁移脚本。
    """
    url = _mysql_dsn()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """以 'online' 模式运行迁移（实际连接数据库执行）。"""
    url = _mysql_dsn()
    config.set_main_option("sqlalchemy.url", url)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
