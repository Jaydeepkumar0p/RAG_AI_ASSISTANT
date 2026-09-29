# ======================================================
# AGENT NODES
# ======================================================

from src.services.intent_classifier import (
    classify_intent,
)

from src.services.query_rewriter import (
    rewrite_query,
)

from src.services.retrieval_service import (
    retrieve_documents,
)

from src.services.retrieval_evaluator import (
    evaluate_retrieval,
)

from src.services.reranker_service import (
    rerank_documents,
)

from src.services.llm_service import (
    llm,
)


# ======================================================
# HELPERS
# ======================================================

def _history_text(
    history: list[dict] | None
) -> str:

    if not history:
        return ""

    parts = []

    for message in history[-6:]:

        role = str(
            message.get(
                "role",
                ""
            )
        )

        content = str(
            message.get(
                "content",
                ""
            )
        )

        parts.append(
            f"{role}: {content}"
        )

    return "\n".join(parts)


def _content(
    response
) -> str:

    value = getattr(
        response,
        "content",
        ""
    )

    if isinstance(
        value,
        str
    ):

        return value.strip()


    if isinstance(
        value,
        list
    ):

        parts = []

        for item in value:

            if isinstance(
                item,
                str
            ):
                parts.append(
                    item
                )

            elif isinstance(
                item,
                dict
            ):

                text = item.get(
                    "text"
                )

                if text:
                    parts.append(
                        str(text)
                    )

        return "".join(
            parts
        ).strip()


    return str(
        value
    ).strip()


# ======================================================
# CLASSIFY
# ======================================================

def classify_intent_node(
    state
):

    intent = classify_intent(

        question=
            state["question"],

        history=
            state.get(
                "history",
                []
            ),
    )

    return {
        "intent": intent
    }


# ======================================================
# QUERY REWRITE
# ======================================================

def rewrite_query_node(
    state
):

    question =
        state["question"]

    history =
        state.get(
            "history",
            []
        )


    # --------------------------------------------------
    # No history
    # --------------------------------------------------

    if not history:

        rewritten =
            rewrite_query(
                question
            )

        return {
            "rewritten_query":
                rewritten
        }


    # --------------------------------------------------
    # History-aware rewrite
    # --------------------------------------------------

    history_text =
        _history_text(
            history
        )


    prompt = f"""
You are a search query rewriting assistant.

Rewrite the latest user question into a
standalone query for a document vector database.

Use conversation history to resolve references
such as:

- it
- this
- that
- they
- them
- which one
- previous answer
- previous topic

Do not answer the question.

Return ONLY the rewritten search query.

Conversation History:
{history_text}

Current Question:
{question}

Rewritten Query:
"""


    response =
        llm.invoke(
            prompt
        )


    rewritten =
        _content(
            response
        )


    if not rewritten:
        rewritten = question


    return {
        "rewritten_query":
            rewritten
    }


# ======================================================
# RETRIEVE DOCUMENTS
# ======================================================

def retrieve_documents_node(
    state
):

    query =
        state.get(
            "rewritten_query"
        ) or state[
            "question"
        ]

    user_id =
        state[
            "user_id"
        ]


    results =
        retrieve_documents(

            query=query,

            user_id=user_id,

            limit=8,
        )


    if not results:

        return {

            "context": "",

            "sources": [],

            "reranker_scores": [],

            "retrieval_attempted":
                True,
        }


    reranked =
        rerank_documents(

            query=query,

            results=results,

            top_k=3,
        )


    context_parts = []

    sources = []

    scores = []


    for item in reranked:

        result =
            item["result"]

        score =
            float(
                item["score"]
            )

        payload =
            result.payload or {}


        text = str(
            payload.get(
                "text",
                ""
            )
        ).strip()


        if text:

            context_parts.append(
                text
            )


        sources.append(
            {
                "filename":
                    payload.get(
                        "filename"
                    ),

                "page":
                    payload.get(
                        "page"
                    ),

                "document_id":
                    payload.get(
                        "document_id"
                    ),
            }
        )


        scores.append(
            score
        )


    return {

        "context":
            "\n\n".join(
                context_parts
            ),

        "sources":
            sources,

        "reranker_scores":
            scores,

        "retrieval_attempted":
            True,
    }


# ======================================================
# EVALUATE RETRIEVAL
# ======================================================

def evaluate_retrieval_node(
    state
):

    context =
        state.get(
            "context",
            ""
        )


    if not context:

        return {
            "retrieval_relevant":
                False
        }


    relevant =
        evaluate_retrieval(

            question=
                state["question"],

            context=
                context,
        )


    return {
        "retrieval_relevant":
            bool(relevant)
    }


# ======================================================
# RAG QA
# ======================================================

def generate_answer_node(
    state
):

    question =
        state[
            "question"
        ]

    context =
        state.get(
            "context",
            ""
        )

    history =
        state.get(
            "history",
            []
        )


    if not context:

        return {

            "answer":
                "I could not find enough relevant information in your uploaded documents to answer that question.",

            "sources":
                [],

        }


    history_text =
        _history_text(
            history
        )


    prompt = f"""
You are an AI Study Assistant.

Answer the user's document-related question
using the provided document context.

Conversation History:
{history_text}

Document Context:
{context}

Current User Question:
{question}

Rules:

- Use document context as the source of truth.
- Do not invent facts about the document.
- Conversation history may resolve references.
- Do not claim information is in the document if it isn't.
- Be clear and concise.

Answer:
"""


    response =
        llm.invoke(
            prompt
        )


    return {

        "answer":
            _content(
                response
            )
    }


# ======================================================
# GENERAL AI
# ======================================================

def general_answer_node(
    state
):

    question =
        state[
            "question"
        ]

    history =
        state.get(
            "history",
            []
        )


    history_text =
        _history_text(
            history
        )


    prompt = f"""
You are a general-purpose AI Study Assistant.

Answer the user's question directly.

You can answer:

- general knowledge
- technical concepts
- AI/ML
- databases
- networking
- cloud
- interview questions
- system design
- software engineering
- career questions
- explanations
- mathematics
- science
- other reasonable questions

Conversation history is available so you can
understand references such as:

- it
- this
- that
- the previous answer
- the above
- that algorithm

Do not claim that information came from the
user's documents unless document context was
actually provided.

If the user asks a technical explanation,
make the explanation structured and practical.

If the user asks a system-design question,
include architecture and trade-offs.

Conversation History:
{history_text}

Current User Question:
{question}

Answer:
"""


    response =
        llm.invoke(
            prompt
        )


    return {

        "answer":
            _content(
                response
            ),

        "sources":
            [],

        "retrieval_relevant":
            None,

        "rewritten_query":
            None,
    }


# ======================================================
# CODING / DSA
# ======================================================

def coding_answer_node(
    state
):

    question =
        state[
            "question"
        ]

    history =
        state.get(
            "history",
            []
        )


    history_text =
        _history_text(
            history
        )


    prompt = f"""
You are an expert software engineer,
DSA instructor, competitive programmer,
debugging expert, and system-design mentor.

Solve the user's coding or technical implementation
request completely.

IMPORTANT:

When the user asks for an algorithm or DSA problem,
use this structure:

## Problem Understanding

Explain the problem.

## Approach

Explain the optimized approach.

## Algorithm

Give numbered steps.

## Code

Provide COMPLETE runnable code.

## Example

Input:
...

Output:
...

## Complexity

Time Complexity:
O(...)

Space Complexity:
O(...)

## Edge Cases

Explain important edge cases.

## Why It Works

Explain correctness and reasoning.

For debugging requests:

## Problem

## Root Cause

## Fixed Code

Provide the COMPLETE corrected code.

## Explanation

Explain the fix.

For project implementation requests:

Start with:

## File Structure

```text
project/
├── ...
└── ...
