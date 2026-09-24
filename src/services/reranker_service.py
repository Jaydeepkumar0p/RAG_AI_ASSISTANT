# ======================================================
# RERANKER SERVICE
# ======================================================
#
# Lightweight reranker for Render Free deployment.
#
# The previous implementation used:
#
#     from sentence_transformers import CrossEncoder
#
# and:
#
#     CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
#
# That loads a PyTorch/Sentence-Transformers model and
# can push a 512 MB Render instance over its memory limit.
#
# This implementation keeps the same public function:
#
#     rerank_documents(query, results, top_k)
#
# so the rest of the RAG / LangGraph pipeline can continue
# using the same interface.
# ======================================================

import re
from typing import Any


# ======================================================
# TOKENIZATION
# ======================================================

def _tokenize(text: str) -> set[str]:
    """
    Convert text into a set of normalized word tokens.
    """

    if not text:
        return set()

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


# ======================================================
# RELEVANCE SCORE
# ======================================================

def _relevance_score(
    query_tokens: set[str],
    document_tokens: set[str]
) -> float:
    """
    Calculate a simple lexical relevance score.

    Score is based on how many query terms occur
    in the candidate document.

    Returns a value between 0.0 and 1.0.
    """

    if not query_tokens:
        return 0.0

    if not document_tokens:
        return 0.0

    matching_tokens = (
        query_tokens & document_tokens
    )

    return (
        len(matching_tokens)
        / len(query_tokens)
    )


# ======================================================
# RERANK DOCUMENTS
# ======================================================

def rerank_documents(
    query: str,
    results: list[Any],
    top_k: int = 3
):
    """
    Rerank Qdrant retrieval results.

    Parameters
    ----------
    query:
        User/search query.

    results:
        List of Qdrant ScoredPoint objects.

    top_k:
        Number of results to return.

    Returns
    -------
    list:
        Same structure expected by the existing pipeline:

        [
            {
                "score": float,
                "result": qdrant_result
            }
        ]
    """

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if not results:
        return []

    if top_k <= 0:
        return []

    # --------------------------------------------------
    # Tokenize query
    # --------------------------------------------------

    query_tokens = _tokenize(query)

    ranked = []

    # --------------------------------------------------
    # Score every candidate
    # --------------------------------------------------

    for result in results:

        # ----------------------------------------------
        # Safely get Qdrant payload
        # ----------------------------------------------

        payload = getattr(
            result,
            "payload",
            None
        )

        if not isinstance(
            payload,
            dict
        ):
            payload = {}

        # ----------------------------------------------
        # Get document text
        # ----------------------------------------------

        text = payload.get(
            "text",
            ""
        )

        if not isinstance(
            text,
            str
        ):
            text = str(text)

        # ----------------------------------------------
        # Calculate lexical relevance
        # ----------------------------------------------

        document_tokens = _tokenize(
            text
        )

        score = _relevance_score(
            query_tokens,
            document_tokens
        )

        # ----------------------------------------------
        # Keep original Qdrant result
        # ----------------------------------------------

        ranked.append(
            (
                float(score),
                result
            )
        )

    # --------------------------------------------------
    # Sort highest relevance first
    # --------------------------------------------------

    ranked.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # --------------------------------------------------
    # Return top-k results
    # --------------------------------------------------

    return [
        {
            "score": score,
            "result": result
        }
        for score, result
        in ranked[:top_k]
    ]


# ======================================================
# OPTIONAL COMPATIBILITY FUNCTION
# ======================================================

def rerank(
    query: str,
    results: list[Any],
    top_k: int = 3
):
    """
    Compatibility wrapper in case another service
    imports `rerank()` instead of `rerank_documents()`.
    """

    return rerank_documents(
        query=query,
        results=results,
        top_k=top_k
    )
