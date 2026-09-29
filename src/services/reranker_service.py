# ======================================================
# LIGHTWEIGHT RERANKER
# ======================================================
#
# This reranker DOES NOT use an LLM.
#
# It combines:
#
# 1. Qdrant semantic similarity score
# 2. Lexical token overlap
#
# Final score:
#
#     0.75 * semantic_score
#   + 0.25 * lexical_score
#
# This keeps the service lightweight and avoids
# loading another ML model such as CrossEncoder.
#
# ======================================================

import re
from typing import Any


# ======================================================
# TOKENIZATION
# ======================================================

def _tokens(text: str) -> set[str]:
    """
    Convert text into a normalized set of tokens.

    Example:

        "Python FastAPI API"
            ↓
        {"python", "fastapi", "api"}
    """

    words = re.findall(
        r"[a-zA-Z0-9_]+",
        str(text or "").lower(),
    )

    return {
        word
        for word in words
        if len(word) > 1
    }


# ======================================================
# LEXICAL OVERLAP SCORE
# ======================================================

def _overlap_score(
    query: str,
    document: str,
) -> float:
    """
    Calculate the percentage of query tokens that
    also appear in the document.

    Example:

        query:
            "python fastapi"

        document:
            "This project uses Python and FastAPI"

        score:
            1.0
    """

    query_tokens = _tokens(
        query
    )

    document_tokens = _tokens(
        document
    )

    if not query_tokens:
        return 0.0

    overlap = len(
        query_tokens
        &
        document_tokens
    )

    return (
        overlap /
        len(query_tokens)
    )


# ======================================================
# QDRANT SEMANTIC SCORE
# ======================================================

def _qdrant_score(
    result: Any,
) -> float:
    """
    Safely extract the similarity score returned
    by Qdrant.
    """

    score = getattr(
        result,
        "score",
        0.0,
    )

    try:
        return float(
            score
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


# ======================================================
# RERANK DOCUMENTS
# ======================================================

def rerank_documents(
    query: str,
    results: list,
    top_k: int = 3,
) -> list[dict]:
    """
    Rerank retrieved Qdrant results.

    Parameters
    ----------
    query:
        User's search query.

    results:
        Qdrant search results.

    top_k:
        Number of final documents to return.

    Returns
    -------
    list[dict]

    Example:

    [
        {
            "score": 0.91,
            "result": qdrant_result
        }
    ]
    """

    if not results:
        return []


    ranked = []


    # ==================================================
    # SCORE EACH RESULT
    # ==================================================

    for result in results:

        payload = (
            getattr(
                result,
                "payload",
                None,
            )
            or {}
        )


        text = str(
            payload.get(
                "text",
                "",
            )
        )


        # ----------------------------------------------
        # Qdrant semantic score
        # ----------------------------------------------

        vector_score = _qdrant_score(
            result
        )


        # ----------------------------------------------
        # Exact / lexical overlap
        # ----------------------------------------------

        lexical_score = _overlap_score(
            query,
            text,
        )


        # ----------------------------------------------
        # Combined score
        #
        # Semantic similarity:
        #       75%
        #
        # Lexical overlap:
        #       25%
        # ----------------------------------------------

        combined_score = (
            0.75 * vector_score
        ) + (
            0.25 * lexical_score
        )


        ranked.append(
            (
                float(combined_score),
                result,
            )
        )


    # ==================================================
    # SORT HIGHEST SCORE FIRST
    # ==================================================

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )


    # ==================================================
    # RETURN TOP-K
    # ==================================================

    output = []

    for score, result in ranked[
        :max(1, int(top_k))
    ]:

        output.append(
            {
                "score": float(score),
                "result": result,
            }
        )


    return output
