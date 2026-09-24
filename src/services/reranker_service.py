from sentence_transformers import CrossEncoder


# ======================================================
# RERANKER MODEL
# ======================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


# ======================================================
# RERANK DOCUMENTS
# ======================================================

def rerank_documents(
    query: str,
    results: list,
    top_k: int = 3
):

    if not results:
        return []

    pairs = []

    for result in results:

        payload = result.payload

        text = payload.get(
            "text",
            ""
        )

        pairs.append(
            (
                query,
                text
            )
        )

    # Cross-encoder scores query/document pairs
    scores = reranker.predict(
        pairs
    )

    ranked = []

    for result, score in zip(
        results,
        scores
    ):

        ranked.append(
            (
                float(score),
                result
            )
        )

    # Highest score first
    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        {
            "score": score,
            "result": result
        }
        for score, result in ranked[:top_k]
    ]