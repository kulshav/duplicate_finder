import time
import numpy as np

from torch.multiprocessing import Pool, set_start_method

from database.models import Item
from services.utils import populate_bulk_embeddings, get_csv_data

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def process_and_insert(args):
    nlp_model, dataset_part = args
    populate_bulk_embeddings(nlp_model, dataset_part, Item)


def upload_data_from_local_path(nlp_model, path: str, column_id: int, is_multiprocess: bool = False, mp_pool: int = 0) -> dict:
    dataset = get_csv_data(data_path=path, column_id=column_id)

    if not is_multiprocess:
        time_spent = populate_bulk_embeddings(
            nlp_model=nlp_model,
            dataset=dataset,
            table_model=Item,
        )

        return {"time_spent": time_spent}

    time_spent = upload_multiprocess(
        nlp_model=nlp_model,
        dataset=dataset,
        mp_pool=mp_pool
    )

    return {"time_spent": time_spent}


def upload_multiprocess(nlp_model, dataset: np.ndarray, mp_pool: int) -> float:

    model = nlp_model
    dataset_parts = np.array_split(dataset, mp_pool)
    time_start = time.time()
    
    with Pool(mp_pool) as pool:
        # Map the function to each part of the dataset in parallel
        pool.map(process_and_insert, [(model, part) for part in dataset_parts])

        # Explicitly close and join the pool to clean up resources
        pool.close()
        pool.join()
    time_spent = time.time() - time_start
    
    return time_spent
