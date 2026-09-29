# ======================================================
# AGENT NODES
# ======================================================

from src.services.query_rewriter import rewrite_query
from src.services.retrieval_service import retrieve_documents
from src.services.retrieval_evaluator import evaluate_retrieval
from src.services.llm_service import llm
from src.services.intent_classifier import classify_intent
from src.services.reranker_service import rerank_documents


# ======================================================
# HELPERS
# ======================================================

def _history_text(history):
    if not history:
        return ""

    lines = []

    for message in history[-6:]:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


def _response_content(response):
    content = getattr(
        response,
        "content",
        ""
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))

        return "".join(parts).strip()

    return str(content).strip()


# ======================================================
# CLASSIFY INTENT NODE
# ======================================================

def classify_intent_node(state):
    question = state["question"]
    history = state.get(
        "history",
        []
    )

    # --------------------------------------------------
    # Support both:
    #
    # classify_intent(question)
    #
    # and:
    #
    # classify_intent(question, history)
    # --------------------------------------------------

    try:
        intent = classify_intent(
            question,
            history
        )

    except TypeError:
        intent = classify_intent(
            question
        )

    return {
        "intent": intent
    }


# ======================================================
# REWRITE QUERY NODE
# ======================================================

def rewrite_query_node(state):
    question = state["question"]

    history = state.get(
        "history",
        []
    )

    # --------------------------------------------------
    # No history
    # --------------------------------------------------

    if not history:
        rewritten_query = rewrite_query(
            question
        )

        return {
            "rewritten_query": (
                rewritten_query
                or question
            )
        }

    # --------------------------------------------------
    # Build history
    # --------------------------------------------------

    history_text = _history_text(
        history
    )

    # --------------------------------------------------
    # Rewrite prompt
    # --------------------------------------------------

    prompt = f"""
You are a search query rewriting assistant.

Rewrite the user's latest question into
a concise standalone search query for a
document vector database.

Use the conversation history only to resolve
references such as:

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

Current User Question:
{question}

Rewritten Query:
"""

    response = llm.invoke(
        prompt
    )

    rewritten_query = _response_content(
        response
    )

    if not rewritten_query:
        rewritten_query = question

    return {
        "rewritten_query": rewritten_query
    }


# ======================================================
# RETRIEVE + RERANK NODE
# ======================================================

def retrieve_documents_node(state):
    rewritten_query = (
        state.get("rewritten_query")
        or state["question"]
    )

    user_id = state[
        "user_id"
    ]

    # --------------------------------------------------
    # Retrieve candidates
    # --------------------------------------------------

    results = retrieve_documents(
        query=rewritten_query,
        user_id=user_id,
        limit=8
    )

    # --------------------------------------------------
    # No results
    # --------------------------------------------------

    if not results:
        return {
            "context": "",
            "sources": [],
            "reranker_scores": [],
            "retrieval_attempted": True,
        }

    # --------------------------------------------------
    # Rerank
    # --------------------------------------------------

    reranked_results = rerank_documents(
        query=rewritten_query,
        results=results,
        top_k=3
    )

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context_parts = []
    sources = []
    reranker_scores = []

    for item in reranked_results:
        result = item["result"]

        score = float(
            item["score"]
        )

        payload = (
            result.payload
            or {}
        )

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
                "filename": payload.get(
                    "filename"
                ),
                "page": payload.get(
                    "page"
                ),
                "document_id": payload.get(
                    "document_id"
                ),
            }
        )

        reranker_scores.append(
            score
        )

    context = "\n\n".join(
        context_parts
    )

    return {
        "context": context,
        "sources": sources,
        "reranker_scores": reranker_scores,
        "retrieval_attempted": True,
    }


# ======================================================
# RETRIEVAL EVALUATION NODE
# ======================================================

def evaluate_retrieval_node(state):
    context = state.get(
        "context",
        ""
    )

    if not context:
        return {
            "retrieval_relevant": False
        }

    is_relevant = evaluate_retrieval(
        question=state["question"],
        context=context
    )

    return {
        "retrieval_relevant": bool(
            is_relevant
        )
    }


# ======================================================
# RAG QA ANSWER NODE
# ======================================================

def generate_answer_node(state):
    question = state[
        "question"
    ]

    context = state.get(
        "context",
        ""
    )

    history = state.get(
        "history",
        []
    )

    history_text = _history_text(
        history
    )

    # --------------------------------------------------
    # No document context
    # --------------------------------------------------

    if not context:
        return {
            "answer": (
                "I could not find enough "
                "relevant information in "
                "your uploaded documents "
                "to answer that question."
            ),
            "sources": [],
        }

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are an AI Study Assistant.

Answer the user's document-related question
using the supplied document context.

Conversation History:
{history_text}

Document Context:
{context}

Current User Question:
{question}

Rules:

- Use the document context as the source of truth.
- Do not invent document facts.
- Do not claim information is present if it is not present.
- Use conversation history only to resolve references.
- Give a clear and useful answer.

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        )
    }


# ======================================================
# GENERAL AI NODE
# ======================================================

def general_answer_node(state):
    question = state[
        "question"
    ]

    history = state.get(
        "history",
        []
    )

    history_text = _history_text(
        history
    )

    prompt = f"""
You are a general-purpose AI Study Assistant.

Answer the user's question directly.

You can answer:

- general knowledge
- technical concepts
- software engineering
- AI and machine learning
- databases
- networking
- cloud
- system design
- interview questions
- mathematics
- science
- career questions
- explanations
- other reasonable questions

Use conversation history when needed to
understand references such as:

- it
- this
- that
- the above
- the previous answer

Do not claim that information came from
the user's uploaded documents unless
document context was explicitly provided.

Make technical explanations structured
and practical.

Conversation History:
{history_text}

Current User Question:
{question}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        ),
        "sources": [],
        "retrieval_relevant": None,
        "rewritten_query": None,
    }


# ======================================================
# CODING / DSA NODE
# ======================================================

def coding_answer_node(state):
    question = state[
        "question"
    ]

    history = state.get(
        "history",
        []
    )

    history_text = _history_text(
        history
    )

    prompt = f"""
You are an expert software engineer,
DSA instructor, competitive programmer,
debugging expert, and system-design mentor.

Solve the user's programming or technical
implementation request completely.

For DSA / algorithm questions use:

## Problem Understanding

Explain the problem.

## Approach

Explain the optimized approach.

## Algorithm

1. Step 1
2. Step 2
3. Step 3

## Code

Provide complete runnable code.

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

Explain correctness.

For debugging requests use:

## Problem

## Root Cause

## Fixed Code

Provide the complete corrected code.

## Explanation

Explain the correction.

For project implementation requests:

First provide:

## File Structure

Then provide every important file.

Use:

### File: `src/example/file.py`

```python
complete code
