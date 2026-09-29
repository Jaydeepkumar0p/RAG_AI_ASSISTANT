import re

from typing import Any


# ======================================================
# TOKENIZATION
# ======================================================

def _tokens(
    text: str,
) -> set[str]:
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
# LEXICAL OVERLAP
# ======================================================

def _overlap_score(
    query: str,
    document: str,
) -> float:
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
        overlap
        /
        len(query_tokens)
    )


# ======================================================
# QDRANT SCORE
# ======================================================

def _qdrant_score(
    result: Any,
) -> float:
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
# RERANK
# ======================================================

def rerank_documents(
    query: str,
    results: list,
    top_k: int = 3,
) -> list[dict]:

    if not results:
        return []

    ranked = []

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

        semantic_score = (
            _qdrant_score(
                result
            )
        )

        lexical_score = (
            _overlap_score(
                query,
                text,
            )
        )

        combined_score = (
            0.75
            *
            semantic_score
        ) + (
            0.25
            *
            lexical_score
        )

        ranked.append(
            (
                float(
                    combined_score
                ),
                result,
            )
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    safe_top_k = max(
        1,
        int(top_k),
    )

    return [
        {
            "score": float(score),
            "result": result,
        }
        for score, result
        in ranked[:safe_top_k]
    ]
