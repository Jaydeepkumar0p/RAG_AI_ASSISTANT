# ======================================================
# LIGHTWEIGHT RERANKER
# ======================================================
#
# No sentence-transformers dependency.
#
# Uses:
# 1. Qdrant similarity score
# 2. Lexical token overlap
#
# This keeps deployment lightweight and avoids loading
# another large ML model on Render.
# ======================================================

import re


# ======================================================
# TOKENIZE
# ======================================================

def _tokens(
    text: str
) -> set[str]:

    words =
        re.findall(
            r"[a-zA-Z0-9_]+",
            str(
                text or ""
            ).lower()
        )

    return {
        word
        for word in words
        if len(word) > 1
    }


# ======================================================
# LEXICAL OVERLAP
# ======================================================

def _overlap_score(
    query: str,
    document: str
) -> float:

    query_tokens =
        _tokens(
            query
        )

    document_tokens =
        _tokens(
            document
        )


    if not query_tokens:
        return 0.0


    overlap =
        len(
            query_tokens
            &
            document_tokens
        )


    return (
        overlap /
        len(
            query_tokens
        )
    )


# ======================================================
# BASE SCORE
# ======================================================

def _qdrant_score(
    result
) -> float:

    score =
        getattr(
            result,
            "score",
            0.0
        )


    try:

        return float(
            score
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0


# ======================================================
# RERANK
# ======================================================

def rerank_documents(
    query: str,
    results: list,
    top_k: int = 3,
):

    if not results:
        return []


    ranked = []


    for result in results:

        payload =
            result.payload or {}


        text =
            str(
                payload.get(
                    "text",
                    ""
                )
            )


        vector_score =
            _qdrant_score(
                result
            )


        lexical_score =
            _overlap_score(
                query,
                text
            )


        # ------------------------------------------------
        # Combined score
        #
        # Qdrant semantic score gets higher weight.
        # Lexical overlap helps exact technical terms.
        # ------------------------------------------------

        combined_score = (
            (
                0.75 *
                vector_score
            )
            +
            (
                0.25 *
                lexical_score
            )
        )


        ranked.append(
            (
                combined_score,
                result
            )
        )


    # ----------------------------------------------------
    # Highest first
    # ----------------------------------------------------

    ranked.sort(
        key=lambda item:
            item[0],

        reverse=True
    )


    return [

        {
            "score":
                float(score),

            "result":
                result,
        }

        for score, result
        in ranked[
            :top_k
        ]
    ]
