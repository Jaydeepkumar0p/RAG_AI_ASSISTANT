from src.services.intent_classifier import classify_intent
from src.services.query_rewriter import rewrite_query
from src.services.retrieval_service import retrieve_documents
from src.services.retrieval_evaluator import evaluate_retrieval
from src.services.reranker_service import rerank_documents
from src.services.llm_service import llm


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
                    parts.append(
                        str(text)
                    )

        return "".join(parts).strip()

    return str(content).strip()


# ======================================================
# CLASSIFY INTENT
# ======================================================

def classify_intent_node(state):
    question = state["question"]

    history = state.get(
        "history",
        []
    )

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
# QUERY REWRITER
# ======================================================

def rewrite_query_node(state):
    question = state["question"]

    history = state.get(
        "history",
        []
    )

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

    history_text = _history_text(
        history
    )

    prompt = (
        "You are a search query rewriting assistant.\n\n"
        "Rewrite the latest user question into a "
        "concise standalone search query for a "
        "document vector database.\n\n"
        "Use conversation history only to resolve "
        "references such as it, this, that, they, "
        "them, which one, previous answer, or "
        "previous topic.\n\n"
        "Do not answer the question.\n"
        "Return ONLY the rewritten search query.\n\n"
        "Conversation History:\n"
        + history_text
        + "\n\nCurrent User Question:\n"
        + question
        + "\n\nRewritten Query:\n"
    )

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
# RETRIEVE DOCUMENTS
# ======================================================

def retrieve_documents_node(state):
    query = (
        state.get("rewritten_query")
        or state["question"]
    )

    user_id = state["user_id"]

    results = retrieve_documents(
        query=query,
        user_id=user_id,
        limit=8
    )

    if not results:
        return {
            "context": "",
            "sources": [],
            "reranker_scores": [],
            "retrieval_attempted": True
        }

    reranked_results = rerank_documents(
        query=query,
        results=results,
        top_k=3
    )

    context_parts = []
    sources = []
    reranker_scores = []

    for item in reranked_results:
        result = item["result"]

        score = float(
            item["score"]
        )

        payload = (
            getattr(
                result,
                "payload",
                None
            )
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
                )
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
        "retrieval_attempted": True
    }


# ======================================================
# EVALUATE RETRIEVAL
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
# RAG / DOCUMENT QA
# ======================================================

def generate_answer_node(state):
    question = state["question"]

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

    if not context:
        return {
            "answer": (
                "I could not find enough relevant "
                "information in your uploaded documents "
                "to answer that question."
            ),
            "sources": []
        }

    prompt = (
        "You are an AI Study Assistant.\n\n"
        "Answer the user's document-related question "
        "using the supplied document context.\n\n"
        "Conversation History:\n"
        + history_text
        + "\n\nDocument Context:\n"
        + context
        + "\n\nCurrent User Question:\n"
        + question
        + "\n\n"
        "Rules:\n"
        "- Use the document context as the source of truth.\n"
        "- Do not invent document facts.\n"
        "- Do not claim something is present if it is not.\n"
        "- Use conversation history to resolve references.\n"
        "- Give a clear and useful answer.\n\n"
        "Answer:\n"
    )

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        )
    }


# ======================================================
# GENERAL AI
# ======================================================

def general_answer_node(state):
    question = state["question"]

    history = state.get(
        "history",
        []
    )

    history_text = _history_text(
        history
    )

    prompt = (
        "You are a general-purpose AI Study Assistant.\n\n"
        "Answer the user's question directly.\n\n"
        "You can answer:\n"
        "- general knowledge\n"
        "- technical concepts\n"
        "- software engineering\n"
        "- AI and machine learning\n"
        "- databases\n"
        "- networking\n"
        "- cloud\n"
        "- system design\n"
        "- interview questions\n"
        "- mathematics\n"
        "- science\n"
        "- career questions\n"
        "- explanations\n"
        "- other reasonable questions\n\n"
        "Use conversation history when needed to understand "
        "references such as it, this, that, the above, "
        "or the previous answer.\n\n"
        "Do not claim information came from the user's "
        "uploaded documents unless document context was "
        "actually supplied.\n\n"
        "Make technical explanations structured and practical.\n\n"
        "Conversation History:\n"
        + history_text
        + "\n\nCurrent User Question:\n"
        + question
        + "\n\nAnswer:\n"
    )

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        ),
        "sources": [],
        "retrieval_relevant": None,
        "rewritten_query": None
    }


# ======================================================
# CODING / DSA / DEBUGGING
# ======================================================

def coding_answer_node(state):
    question = state["question"]

    history = state.get(
        "history",
        []
    )

    history_text = _history_text(
        history
    )

    prompt = (
        "You are an expert software engineer, "
        "DSA instructor, competitive programmer, "
        "debugging expert, and system-design mentor.\n\n"
        "Solve the user's programming or technical "
        "implementation request completely.\n\n"

        "For DSA and algorithm questions use:\n\n"

        "## Problem Understanding\n"
        "Explain the problem.\n\n"

        "## Approach\n"
        "Explain the optimized approach.\n\n"

        "## Algorithm\n"
        "Give numbered steps.\n\n"

        "## Code\n"
        "Provide complete runnable code.\n\n"

        "## Example\n"
        "Input:\n"
        "...\n\n"
        "Output:\n"
        "...\n\n"

        "## Complexity\n"
        "Time Complexity: O(...)\n"
        "Space Complexity: O(...)\n\n"

        "## Edge Cases\n"
        "Explain important edge cases.\n\n"

        "## Why It Works\n"
        "Explain the reasoning.\n\n"

        "For debugging requests use:\n\n"

        "## Problem\n"
        "Explain the problem.\n\n"

        "## Root Cause\n"
        "Explain the root cause.\n\n"

        "## Fixed Code\n"
        "Provide the complete corrected code.\n\n"

        "## Explanation\n"
        "Explain the fix.\n\n"

        "For project implementation requests:\n\n"

        "## File Structure\n"
        "List the project files that need to change.\n\n"

        "For every file provide:\n"
        "File: src/path/file.py\n"
        "Then provide the complete code.\n\n"

        "Do not provide placeholders.\n"
        "Do not say 'rest of the code'.\n"
        "Do not omit required files.\n\n"

        "If the user specifies a language, use it.\n"
        "If no language is specified for an algorithm "
        "problem, use Python.\n\n"

        "Conversation History:\n"
        + history_text
        + "\n\nCurrent User Question:\n"
        + question
        + "\n\nAnswer:\n"
    )

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        ),
        "sources": [],
        "retrieval_relevant": None,
        "rewritten_query": None
    }


# ======================================================
# SUMMARY
# ======================================================

def summary_node(state):
    context = state.get(
        "context",
        ""
    )

    if not context:
        return {
            "answer": (
                "I could not find enough relevant "
                "information in the uploaded documents "
                "to create a summary."
            ),
            "sources": []
        }

    question = state["question"]

    prompt = (
        "You are an AI Study Assistant.\n\n"
        "Create a useful summary using only the "
        "supplied document context.\n\n"
        "Document Context:\n"
        + context
        + "\n\nUser Request:\n"
        + question
        + "\n\n"
        "Rules:\n"
        "- Do not invent facts.\n"
        "- Preserve important names.\n"
        "- Preserve dates.\n"
        "- Preserve technologies.\n"
        "- Preserve roles.\n"
        "- Preserve achievements.\n"
        "- Use headings and bullet points when useful.\n\n"
        "Summary:\n"
    )

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        )
    }


# ======================================================
# QUIZ
# ======================================================

def quiz_node(state):
    context = state.get(
        "context",
        ""
    )

    if not context:
        return {
            "answer": (
                "I could not find enough relevant "
                "information in the uploaded documents "
                "to generate a quiz."
            ),
            "sources": []
        }

    question = state["question"]

    prompt = (
        "You are an AI Study Assistant.\n\n"
        "Create a quiz using only the supplied "
        "document context.\n\n"
        "User Request:\n"
        + question
        + "\n\nDocument Context:\n"
        + context
        + "\n\n"
        "Rules:\n"
        "- Create clear multiple-choice questions.\n"
        "- Use exactly four options per question.\n"
        "- Use exactly one correct answer.\n"
        "- Do not invent facts.\n"
        "- Keep every question grounded in the document.\n\n"
        "Format:\n\n"
        "Question 1:\n"
        "<question>\n\n"
        "A. <option>\n"
        "B. <option>\n"
        "C. <option>\n"
        "D. <option>\n\n"
        "Correct Answer:\n"
        "<answer>\n\n"
        "Question 2:\n"
        "...\n\n"
        "Quiz:\n"
    )

    response = llm.invoke(
        prompt
    )

    return {
        "answer": _response_content(
            response
        )
    }


# ======================================================
# RETRY
# ======================================================

def retry_node(state):
    retry_count = int(
        state.get(
            "retry_count",
            0
        )
    )

    return {
        "retry_count": retry_count + 1,
        "context": "",
        "sources": [],
        "reranker_scores": [],
        "retrieval_attempted": False,
        "retrieval_relevant": False
    }


# ======================================================
# REJECT
# ======================================================

def reject_node(state):
    return {
        "answer": (
            "I could not find enough relevant "
            "information in your uploaded documents "
            "to answer that document-specific question."
        ),
        "sources": []
    }
