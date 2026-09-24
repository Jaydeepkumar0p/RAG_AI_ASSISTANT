from fastembed import TextEmbedding

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

from src.database.qdrant import (
    client,
    COLLECTION_NAME
)


_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print("Loading FastEmbed retrieval model...")

        _embeddings = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("FastEmbed retrieval model loaded.")

    return _embeddings


def retrieve_documents(
    query: str,
    user_id: str,
    limit: int = 8
):

    embeddings = get_embeddings()

    # ----------------------------------------------
    # Embed query
    # ----------------------------------------------

    query_vector = list(
        embeddings.embed([query])
    )[0]

    query_vector = (
        query_vector.tolist()
        if hasattr(query_vector, "tolist")
        else list(query_vector)
    )

    # ----------------------------------------------
    # User isolation
    # ----------------------------------------------

    user_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(
                    value=user_id
                )
            )
        ]
    )

    # ----------------------------------------------
    # Qdrant search
    # ----------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=user_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False
    ).points

    return results
