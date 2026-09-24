from qdrant_client import QdrantClient

from src.core.config import settings


# ======================================================
# CONFIG
# ======================================================

QDRANT_URL = (
    settings.QDRANT_URL.strip()
)

QDRANT_API_KEY = (
    settings.QDRANT_API_KEY.strip()
)

COLLECTION_NAME = (
    settings.QDRANT_COLLECTION
)


# ======================================================
# QDRANT CLIENT
# ======================================================

if QDRANT_URL:

    # --------------------------------------------------
    # Production
    # Qdrant Cloud
    # --------------------------------------------------

    client = QdrantClient(

        url=QDRANT_URL,

        api_key=QDRANT_API_KEY
    )

else:

    # --------------------------------------------------
    # Development
    # Local Qdrant
    # --------------------------------------------------

    client = QdrantClient(

        path="./qdrant_data"
    )