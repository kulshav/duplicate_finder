from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base


class DatabaseCore:
    def __init__(self, url: str, echo: bool):

        self.engine = create_engine(url=url, echo=echo)
        self.session = sessionmaker(
            self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        )

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(self.engine)

    def restart_database(self):
        self.drop_tables()
        self.create_tables()


if __name__ == "__main__":
    from configs.settings import db_settings

    db_core = DatabaseCore(url=db_settings.database_url_syncpg, echo=True)
    db_core.create_tables()
