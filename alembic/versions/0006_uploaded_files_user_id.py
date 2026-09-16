"""uploaded_files 属主列（文件撤销的授权依据）。

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-13

说明
----
``uploaded_files`` 新增 ``user_id``：上传者的记忆身份（X-User-Id），
文件撤销端点据此做对象级属主校验（P0 鉴权的配套）。旧行缺省空串，
语义为"无主遗留"——允许任意已验身份用户撤销（file_id 本身 48 位
随机不可猜）。sqlite 侧无 alembic，由运行时自迁移（PRAGMA 检查 +
ALTER TABLE，见 ``extensions/filestore.py``）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级 schema：uploaded_files 补 user_id 列（幂等）。"""
    op.execute(
        sa.text(
            "ALTER TABLE `uploaded_files`"
            " ADD COLUMN `user_id` VARCHAR(64) NOT NULL DEFAULT '' AFTER `module`"
        )
    )


def downgrade() -> None:
    """降级 schema：删除 user_id 列。"""
    op.execute(sa.text("ALTER TABLE `uploaded_files` DROP COLUMN `user_id`"))
