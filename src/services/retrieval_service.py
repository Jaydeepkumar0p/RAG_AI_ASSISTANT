from langchain_huggingface import (
    HuggingFaceEmbeddings
)

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
# EMBEDDINGS
# ======================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ======================================================
# RETRIEVE DOCUMENTS
# ======================================================

def retrieve_documents(
    query: str,
    user_id: str,
    limit: int = 8
):

    query = query.strip()

    if not query:

        return []

    # --------------------------------------------------
    # Query embedding
    # --------------------------------------------------

    query_vector = embeddings.embed_query(
        query
    )

    # --------------------------------------------------
    # User isolation filter
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Qdrant search
    # --------------------------------------------------

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=user_filter,

        limit=limit,

        with_payload=True,

        with_vectors=False

    ).points

    return results