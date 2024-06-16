import time

import pandas as pd
import numpy as np

from database.models import Item, ItemNoEng, ItemLoginom
from database.query import db_query


def get_csv_data(data_path: str, column_id: int) -> np.array:
    dataframe = pd.read_csv(data_path)
    dataset = dataframe.iloc[:, column_id].to_numpy()

    return dataset


# 5 minutes
def populate_database(
    nlp_model,
    dataset: np.ndarray,
    table_model: Item | ItemNoEng | ItemLoginom
    ) -> None:

    for sentence in dataset:
        embedding = nlp_model.encode(sentence)

        db_query.insert_embeddings(
            table_model=table_model,
            sentence=sentence,
            embedding=embedding
        )


def populate_bulk_embeddings(
    nlp_model,
    dataset: np.ndarray,
    table_model: Item | ItemNoEng | ItemLoginom
) -> time:
    start_time = time.time()
    data_to_insert = []

    for sentence in dataset:
        embedding = nlp_model.encode(sentence)
        data_to_insert.append({"sentence": sentence, "embedding": embedding})

    db_query.bulk_insert(
        table_model=table_model,
        data=data_to_insert
    )

    return time.time() - start_time


def truncate_table(table_name: str) -> bool:
    result = db_query.truncate_table(table_name)
    return bool(result)

