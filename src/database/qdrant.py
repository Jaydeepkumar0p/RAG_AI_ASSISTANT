from qdrant_client import QdrantClient
from qdrant_client import models

from src.core.config import settings


# ======================================================
# CONFIG
# ======================================================

QDRANT_URL = (
    settings.QDRANT_URL.strip()
    if settings.QDRANT_URL
    else ""
)

QDRANT_API_KEY = (
    settings.QDRANT_API_KEY.strip()
    if settings.QDRANT_API_KEY
    else ""
)

COLLECTION_NAME = (
    settings.QDRANT_COLLECTION
)


# ======================================================
# CLIENT
# ======================================================

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is not configured"
    )


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ======================================================
# COLLECTION INITIALIZATION
# ======================================================

def initialize_qdrant():
    """
    Ensure collection and required payload indexes exist.
    """

    # --------------------------------------------------
    # Collection
    # --------------------------------------------------

    if not client.collection_exists(
        COLLECTION_NAME
    ):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{COLLECTION_NAME}"
        )

    # --------------------------------------------------
    # user_id index
    # --------------------------------------------------

    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print(
            "Qdrant user_id index ready"
        )

    except Exception as exc:
        print(
            "user_id index already exists "
            "or could not be created:",
            exc,
        )

    # --------------------------------------------------
    # document_id index
    # --------------------------------------------------

    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print(
            "Qdrant document_id index ready"
        )

    except Exception as exc:
        print(
            "document_id index already exists "
            "or could not be created:",
            exc,
        )


# ======================================================
# DELETE DOCUMENT VECTORS
# ======================================================

def delete_document_vectors(
    document_id: str,
    user_id: str,
):
    """
    Delete all Qdrant vectors belonging to one document
    and one authenticated user.
    """

    if not document_id:
        raise ValueError(
            "document_id is required"
        )

    if not user_id:
        raise ValueError(
            "user_id is required"
        )

    # --------------------------------------------------
    # Build a proper Qdrant filter.
    #
    # DO NOT use a plain dict here.
    # --------------------------------------------------

    delete_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(
                    value=document_id
                ),
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(
                    value=user_id
                ),
            ),
        ]
    )

    print(
        "Deleting Qdrant vectors:",
        {
            "document_id": document_id,
            "user_id": user_id,
        },
    )

    # --------------------------------------------------
    # Delete matching points
    # --------------------------------------------------

    operation = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=delete_filter,
        wait=True,
    )

    print(
        "Qdrant delete completed:",
        operation,
    )

    return operation


# ======================================================
# QDRANT HEALTH
# ======================================================

def check_qdrant_connection():
    try:
        client.get_collections()

        print(
            "Qdrant connected successfully"
        )

        return True

    except Exception as exc:
        print(
            "Qdrant connection error:",
            exc,
        )

        return False
