from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from qdrant_client import QdrantClient

from src.database.qdrant import (
    COLLECTION_NAME
)


# ======================================================
# QDRANT
# ======================================================

client = QdrantClient(
    path="./qdrant_data"
)


# ======================================================
# DENSE EMBEDDINGS
# ======================================================

dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ======================================================
# SPARSE EMBEDDINGS
# ======================================================

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)


# ======================================================
# HYBRID STORE
# ======================================================

vector_store = QdrantVectorStore(

    client=client,

    collection_name=COLLECTION_NAME,

    embedding=dense_embeddings,

    sparse_embedding=sparse_embeddings,

    retrieval_mode=RetrievalMode.HYBRID,

    vector_name="dense",

    sparse_vector_name="sparse"
)


# ======================================================
# HYBRID SEARCH
# ======================================================

def hybrid_search(
    query: str,
    limit: int = 8
):

    documents = vector_store.similarity_search(

        query,

        k=limit
    )

    return documents