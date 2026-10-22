"""Alembic 迁移环境：记忆包独立链路（version_table=alembic_version_memory）。

连接参数与基座保持一致：显式 AGENT_BASE_DB_* 环境变量优先，否则回退到
agent_base.core.config.Settings（读取 .env 里的 CHECKPOINTER_MYSQL_*）。
差异只有一处——版本表用单独的名字，这样同一个库里两条迁移链互不干扰：
基座的 alembic_version 停在它自己的 head，记忆包的 alembic_version_memory
停在 0001。迁移本身是手写 SQL（没有 ORM 模型），autogenerate 不可用。
"""

import os
from logging.config import fileConfig
from urllib.parse import quote

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic 的 Config 对象，提供 .ini 中的值。
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

# 与应用表同一库，但版本记录写在自己的表里。
VERSION_TABLE = "alembic_version_memory"


def _mysql_dsn() -> str:
    """组装 MySQL DSN：AGENT_BASE_DB_* > .env 中的 CHECKPOINTER_MYSQL_*。"""
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
        # 兜底：只有环境变量可用时也能跑（例如 CI 里未安装基座的配置层）。
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
    """offline 模式：只生成 SQL，不连库。"""
    url = _mysql_dsn()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """online 模式：连库执行迁移。"""
    url = _mysql_dsn()
    config.set_main_option("sqlalchemy.url", url)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=VERSION_TABLE,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
