import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastembed import TextEmbedding

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
        print("Loading FastEmbed model...")

        _embeddings = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("FastEmbed model loaded.")

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

    # --------------------------------------------------
    # Create collection if it does not exist
    # --------------------------------------------------

    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
        )

    # --------------------------------------------------
    # Ensure user_id index
    # --------------------------------------------------

    try:

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        print("user_id payload index ready")

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

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    # --------------------------------------------------
    # 2. Split PDF
    # --------------------------------------------------

    chunks = splitter.split_documents(documents)

    if not chunks:
        return {
            "chunks": 0
        }

    # --------------------------------------------------
    # 3. Extract valid text
    # --------------------------------------------------

    texts = []
    valid_chunks = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if not text:
            continue

        texts.append(text)
        valid_chunks.append(chunk)

    if not texts:
        return {
            "chunks": 0
        }

    # --------------------------------------------------
    # 4. Load embedding model only when needed
    # --------------------------------------------------

    embeddings = get_embeddings()

    # --------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------

    vectors = list(
        embeddings.embed(texts)
    )

    # --------------------------------------------------
    # 6. Create Qdrant points
    # --------------------------------------------------

    points = []

    for chunk, vector in zip(
        valid_chunks,
        vectors
    ):

        vector_list = (
            vector.tolist()
            if hasattr(vector, "tolist")
            else list(vector)
        )

        point = models.PointStruct(
            id=str(uuid.uuid4()),

            vector=vector_list,

            payload={
                "text": chunk.page_content.strip(),

                "page": chunk.metadata.get("page"),

                "user_id": user_id,

                "document_id": document_id,

                "filename": filename
            }
        )

        points.append(point)

    # --------------------------------------------------
    # 7. Nothing to upload
    # --------------------------------------------------

    if not points:
        return {
            "chunks": 0
        }

    # --------------------------------------------------
    # 8. Upload to Qdrant
    # --------------------------------------------------

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    print(
        f"Uploaded {len(points)} chunks to Qdrant"
    )

    return {
        "chunks": len(points)
    }
