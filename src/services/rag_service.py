import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import models

from src.database.qdrant import (
    client,
    COLLECTION_NAME
)


# ======================================================
# EMBEDDINGS
# ======================================================

_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print("Loading embedding model...")

        _embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        print("Embedding model loaded.")

    return _embeddings


# ======================================================
# TEXT SPLITTER
# ======================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# ======================================================
# CREATE / ENSURE COLLECTION
# ======================================================

def create_collection():

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config=models.VectorParams(

                size=384,

                distance=models.Distance.COSINE
            )
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    try:

        client.create_payload_index(

            collection_name=COLLECTION_NAME,

            field_name="user_id",

            field_schema=(
                models.PayloadSchemaType.KEYWORD
            )
        )

        print(
            "user_id payload index ready"
        )

    except Exception as e:

        print(
            "user_id index already exists "
            "or could not be recreated:",
            e
        )


# ======================================================
# PROCESS PDF
# ======================================================

def process_pdf(

    file_path: str,

    user_id: str,

    document_id: str,

    filename: str

):

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()

    # --------------------------------------------------
    # 2. Split PDF
    # --------------------------------------------------

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:

        return {
            "chunks": 0
        }

    # --------------------------------------------------
    # 3. Load embeddings ONLY when needed
    # --------------------------------------------------

    embeddings = get_embeddings()

    # --------------------------------------------------
    # 4. Create Qdrant points
    # --------------------------------------------------

    points = []

    for chunk in chunks:

        text = (
            chunk.page_content
            .strip()
        )

        if not text:

            continue

        # ----------------------------------------------
        # Generate embedding
        # ----------------------------------------------

        vector = (
            embeddings.embed_query(
                text
            )
        )

        # ----------------------------------------------
        # Create point
        # ----------------------------------------------

        point = models.PointStruct(

            id=str(
                uuid.uuid4()
            ),

            vector=vector,

            payload={

                "text":
                    text,

                "page":
                    chunk.metadata.get(
                        "page"
                    ),

                "user_id":
                    user_id,

                "document_id":
                    document_id,

                "filename":
                    filename
            }
        )

        points.append(
            point
        )

    # --------------------------------------------------
    # 5. No valid content
    # --------------------------------------------------

    if not points:

        return {
            "chunks": 0
        }

    # --------------------------------------------------
    # 6. Upload
    # --------------------------------------------------

    client.upsert(

        collection_name=COLLECTION_NAME,

        points=points,

        wait=True
    )

    print(
        f"Uploaded {len(points)} "
        f"chunks to Qdrant"
    )

    return {
        "chunks": len(points)
    }