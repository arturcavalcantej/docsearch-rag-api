"""add uuid defaults

Revision ID: a7e9c28ddd58
Revises: 9945cf20762f
Create Date: 2025-12-29 20:31:54.891303

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7e9c28ddd58'
down_revision: Union[str, None] = '9945cf20762f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # garante função gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # seta default no id para gerar UUID no banco
    op.execute("ALTER TABLE documents ALTER COLUMN id SET DEFAULT gen_random_uuid()")
    op.execute("ALTER TABLE chunks ALTER COLUMN id SET DEFAULT gen_random_uuid()")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN id DROP DEFAULT")
    op.execute("ALTER TABLE documents ALTER COLUMN id DROP DEFAULT")
