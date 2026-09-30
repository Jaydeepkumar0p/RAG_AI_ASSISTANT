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
# COLLECTION
# ======================================================

def ensure_collection():
    """
    Create collection if it does not exist.
    """

    exists = client.collection_exists(
        COLLECTION_NAME
    )

    if not exists:

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


# ======================================================
# PAYLOAD INDEXES
# ======================================================

def ensure_payload_indexes():
    """
    Ensure all fields used in Qdrant filters have
    payload indexes.

    IMPORTANT:
    document_id is required for deleting vectors
    belonging to a document.
    """

    # --------------------------------------------------
    # user_id
    # --------------------------------------------------

    try:

        client.create_payload_index(
            collection_name=COLLECTION_NAME,

            field_name="user_id",

            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print(
            "Qdrant index ready: user_id"
        )

    except Exception as exc:

        print(
            "user_id index already exists "
            "or could not be created:",
            exc,
        )


    # --------------------------------------------------
    # document_id
    # --------------------------------------------------
    #
    # THIS IS THE IMPORTANT FIX.
    # --------------------------------------------------

    try:

        client.create_payload_index(
            collection_name=COLLECTION_NAME,

            field_name="document_id",

            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print(
            "Qdrant index ready: document_id"
        )

    except Exception as exc:

        print(
            "document_id index already exists "
            "or could not be created:",
            exc,
        )


# ======================================================
# INITIALIZE
# ======================================================

def initialize_qdrant():
    """
    Run this once when the FastAPI application starts.
    """

    ensure_collection()

    ensure_payload_indexes()

    print(
        "Qdrant initialization complete"
    )


# ======================================================
# OPTIONAL HEALTH CHECK
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
