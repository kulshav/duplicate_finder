"""Init

Revision ID: 4d743c2bec5e
Revises: 
Create Date: 2024-06-14 22:57:31.070318

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '4d743c2bec5e'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('itemloginoms',
    sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
    sa.Column('sentence', sa.String(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('itemnoengs',
    sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
    sa.Column('sentence', sa.String(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('items',
    sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
    sa.Column('sentence', sa.String(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('items')
    op.drop_table('itemnoengs')
    op.drop_table('itemloginoms')
    # ### end Alembic commands ###
