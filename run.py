# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI, Response
from starlette.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from services import uploader, utils, similarity


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
app = FastAPI(
    description="FastAPI application for finding duplicates with SBERT nlp-model"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/emb", tags=["Embeddings"])
async def get_embedding(word: str):
    """
    Get embedding for a word.\n

    Args:\n
        word (str): The word, sentence or any str object to get the embedding for.

    Returns:\n
        dict: Dictionary containing the word and its embedding.
    """
    result = model.encode(word)
    return {"word": word, "embedding": result.tolist()}


@app.post("/truncate", tags=["Database"])
async def truncate_table():
    """
    Truncate the Items table.\n

    Returns:\n
        Response: Response with status code 200 if successful.
    """
    result = utils.truncate_table(table_name="Items")
    if result:
        return Response(status_code=200)


@app.post("/upload", tags=["Database"])
async def upload_data_to_database(
    path: str,
    column_id: int,
    is_local: bool = True,
    is_mp: bool = False,
    mp_pool: int | None = None
) -> dict:
    """
    Upload data from a CSV file to the database.\n

    Args:\n
        path (str): Path to the CSV file.
        column_id (int): Column ID to use for data insertion.
        is_local (bool): Whether the file is local or not. Defaults to True.
        is_mp (bool): Whether to use multiprocessing. Defaults to False.
        mp_pool (int | None): Number of processes for multiprocessing. Defaults to None.

    Returns:\n
        dict: Dictionary containing the success message and time spent if successful, or an error message.
    """
    if is_local:
        result = uploader.upload_data_from_local_path(
            nlp_model=model,
            path=path,
            column_id=column_id,
            is_multiprocess=is_mp,
            mp_pool=mp_pool,
        )
        if result:
            return {"success": "OK", "time_spent": result["time_spent"]}
        else:
            return {"error": result["error"]}
    return {"Error": "NotImplemented"}  # Not implemented for non-local csv uploads


@app.get("/similar", tags=["Similarity"])
async def get_most_similar(promt: str, limit: int = 50) -> dict:
    """
    Get the most similar results based on a prompt.

    Args:\n
        promt (str): The prompt string.\n
        limit (int): Maximum number of results to return. Defaults to 50.\n

    Returns:\n
        dict: Dictionary containing the prompt and the list of most similar results.
    """
    result = similarity.get_most_similar_results(
        nlp_model=model,
        promt=promt,
        limit=limit
    )
    return {"promt": promt, "result": result}


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
