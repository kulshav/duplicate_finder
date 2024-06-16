from typing import NoReturn

import numpy as np

from loguru import logger

from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError

from configs.settings import db_settings
from database.core import DatabaseCore

from database.models import ItemLoginom, Item, ItemNoEng


class DatabaseQuery(DatabaseCore):
    def __init__(self, url: str, echo: bool):
        super().__init__(url=url, echo=echo)

    def _commit_query(self, query):
        try:
            with self.session() as session:
                res = session.execute(query)
                session.commit()

                return res
        except DBAPIError as error:
            logger.error(error)

    def _execute_query(self, query):
        try:
            with self.session() as session:
                res = session.execute(query)
                return res
        except DBAPIError as error:
            logger.error(error)

    def insert_embeddings(
        self, 
        table_model: ItemNoEng | ItemLoginom | Item | str,
        sentence: str,
        embedding: np.array
        ) -> NoReturn:

        self._commit_query(
            insert(table_model)
            .values(
                sentence=sentence,
                embedding=embedding
            )
        )

    def bulk_insert(
            self,
            table_model,
            data: list[dict]
    ) -> None:
        try:
            with self.session() as session:
                session.bulk_insert_mappings(table_model, data)
                session.commit()
        except DBAPIError as error:
            logger.error(error)

    def select_cosine_similarity(
        self,
        embedding: np.ndarray,
        limit: int | None,
    ) -> list:

        result = self._execute_query(
            select(
                Item.sentence,
                1 - Item.embedding.cosine_distance(embedding),
            )
            .order_by(Item.embedding.cosine_distance(embedding))
            .limit(limit)
        )
        
        return result.all()

    def fetch_all(self, table_model) -> list:
        result = self._execute_query(
            select(
                table_model.sentence,
                table_model.embedding
                )
            )
        return result.all()
    
    def truncate_table(self, table_model):
        result = self._commit_query(text(f"TRUNCATE TABLE {str(table_model)}"))
        return result


db_query = DatabaseQuery(url=db_settings.database_url_syncpg, echo=db_settings.db_echo)
