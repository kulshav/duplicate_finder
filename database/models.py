import numpy as np

from pgvector.sqlalchemy import Vector

from sqlalchemy import BIGINT
from sqlalchemy.orm import mapped_column, Mapped, DeclarativeBase, declared_attr


class Base(DeclarativeBase):

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return f"{cls.__name__.lower()}s"


class Item(Base):
    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    sentence: Mapped[str]
    embedding: Mapped[np.array] = mapped_column(Vector)


class ItemNoEng(Base):
    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    sentence: Mapped[str]
    embedding: Mapped[np.array] = mapped_column(Vector)


class ItemLoginom(Base):
    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    sentence: Mapped[str]
    embedding: Mapped[np.array] = mapped_column(Vector)
