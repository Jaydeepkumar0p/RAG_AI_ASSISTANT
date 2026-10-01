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


# ======================================================
# EMBEDDING MODEL
# ======================================================

_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print(
            "Loading FastEmbed retrieval model..."
        )

        _embeddings = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print(
            "FastEmbed retrieval model loaded."
        )

    return _embeddings


# ======================================================
# QUERY EMBEDDING
# ======================================================

def create_query_embedding(
    query: str
):

    embeddings = get_embeddings()

    vector = list(
        embeddings.embed(
            [query]
        )
    )[0]

    if hasattr(
        vector,
        "tolist"
    ):
        return vector.tolist()

    return list(vector)


# ======================================================
# USER FILTER
# ======================================================

def create_user_filter(
    user_id: str
):

    return Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(
                    value=user_id
                )
            )
        ]
    )


# ======================================================
# RETRIEVE DOCUMENTS
# ======================================================

def retrieve_documents(
    query: str,
    user_id: str,
    limit: int = 12
):

    query = query.strip()

    if not query:
        return []

    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    query_vector = create_query_embedding(
        query
    )

    # --------------------------------------------------
    # User isolation
    # --------------------------------------------------

    user_filter = create_user_filter(
        user_id
    )

    # --------------------------------------------------
    # Qdrant retrieval
    # --------------------------------------------------

    response = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=user_filter,

        limit=limit,

        with_payload=True,

        with_vectors=False
    )

    results = response.points

    # --------------------------------------------------
    # Debug logging
    # --------------------------------------------------

    print(
        f"\n[RETRIEVAL] Query: {query}"
    )

    print(
        f"[RETRIEVAL] Results: {len(results)}"
    )

    for index, result in enumerate(
        results
    ):

        payload = result.payload or {}

        text = payload.get(
            "text",
            ""
        )

        print(
            f"\n--- Result {index + 1} ---"
        )

        print(
            f"Score: {result.score}"
        )

        print(
            f"Filename: "
            f"{payload.get('filename')}"
        )

        print(
            f"Page: "
            f"{payload.get('page')}"
        )

        print(
            f"Text: "
            f"{text[:300]}"
        )

    return results
