from database.query import db_query


def get_most_similar_results(
        nlp_model,
        promt: str,
        limit: int = 50,
) -> list:

    promt_embedding = nlp_model.encode(promt)

    query_result = db_query.select_cosine_similarity(
        embedding=promt_embedding,
        limit=limit
    )

    # Transforming the list of tuples into a list of dictionaries
    result_list = [{"name": item[0], "similarity": item[1]} for item in query_result]

    return result_list
