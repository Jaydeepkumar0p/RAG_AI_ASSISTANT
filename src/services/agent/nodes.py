from src.services.query_rewriter import rewrite_query
from src.services.retrieval_service import retrieve_documents
from src.services.retrieval_evaluator import evaluate_retrieval
from src.services.llm_service import llm
from src.services.intent_classifier import classify_intent

from src.services.reranker_service import (
    rerank_documents
)

from src.services.multi_query_service import (
    generate_search_queries
)


# ======================================================
# CLASSIFY INTENT NODE
# ======================================================

def classify_intent_node(state):

    intent = classify_intent(
        state["question"]
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

    history_text = ""

    for message in history[-6:]:

        history_text += (
            f'{message.get("role", "")}: '
            f'{message.get("content", "")}\n'
        )

    # --------------------------------------------------
    # No history
    # --------------------------------------------------

    if not history_text:

        rewritten_query = rewrite_query(
            question
        )

        return {
            "rewritten_query": rewritten_query
        }

    # --------------------------------------------------
    # Context-aware rewriting
    # --------------------------------------------------

    prompt = f"""
You are a search query rewriting assistant.

Rewrite the user's latest question into a
clear standalone search query for a vector database.

Use the conversation history to resolve references
such as:

- it
- this
- that
- they
- previous answer
- previous topic

Do not answer the question.

Return ONLY the rewritten search query.

Conversation History:

{history_text}

Current User Question:

{question}

Rewritten Search Query:
"""

    response = llm.invoke(
        prompt
    )

    rewritten_query = response.content.strip()

    return {
        "rewritten_query": rewritten_query
    }


# ======================================================
# MULTI-QUERY NODE
# ======================================================

def multi_query_node(state):

    question = state["question"]

    history = state.get(
        "history",
        []
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    # --------------------------------------------------
    # Generate multiple search queries
    # --------------------------------------------------

    queries = generate_search_queries(

        question=question,

        history=history
    )

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not queries:

        queries = [
            state.get(
                "rewritten_query",
                question
            )
        ]

    # --------------------------------------------------
    # During retry, make sure we don't lose
    # the current rewritten query
    # --------------------------------------------------

    rewritten_query = queries[0]

    return {

        "search_queries": queries,

        "rewritten_query": rewritten_query,

        "retry_count": retry_count
    }


# ======================================================
# RETRIEVE + RERANK NODE
# ======================================================

def retrieve_documents_node(state):

    search_queries = state.get(
        "search_queries",
        []
    )

    # --------------------------------------------------
    # Fallback to rewritten query
    # --------------------------------------------------

    if not search_queries:

        search_queries = [
            state.get(
                "rewritten_query",
                state["question"]
            )
        ]

    user_id = state["user_id"]

    all_results = []

    seen_ids = set()

    # ==================================================
    # RETRIEVE FOR EACH QUERY
    # ==================================================

    for query in search_queries:

        results = retrieve_documents(

            query=query,

            user_id=user_id,

            limit=5
        )

        for result in results:

            point_id = str(
                result.id
            )

            # --------------------------------------------------
            # Remove duplicate Qdrant points
            # --------------------------------------------------

            if point_id in seen_ids:
                continue

            seen_ids.add(
                point_id
            )

            all_results.append(
                result
            )

    # ==================================================
    # NO RESULTS
    # ==================================================

    if not all_results:

        return {

            "context": "",

            "sources": [],

            "reranker_scores": []
        }

    # ==================================================
    # RERANK
    # ==================================================

    reranked_results = rerank_documents(

        query=state.get(
            "rewritten_query",
            state["question"]
        ),

        results=all_results,

        top_k=3
    )

    # ==================================================
    # BUILD CONTEXT
    # ==================================================

    context_parts = []

    sources = []

    reranker_scores = []

    seen_sources = set()

    for item in reranked_results:

        result = item["result"]

        score = item["score"]

        payload = result.payload

        text = payload.get(
            "text",
            ""
        )

        # --------------------------------------------------
        # Add text
        # --------------------------------------------------

        if text:

            context_parts.append(
                text
            )

        # --------------------------------------------------
        # Source
        # --------------------------------------------------

        filename = payload.get(
            "filename"
        )

        page = payload.get(
            "page"
        )

        document_id = payload.get(
            "document_id"
        )

        source_key = (
            str(document_id),
            str(page),
            str(filename)
        )

        if source_key not in seen_sources:

            seen_sources.add(
                source_key
            )

            sources.append({

                "filename": filename,

                "page": page,

                "document_id": document_id
            })

        # --------------------------------------------------
        # Reranker score
        # --------------------------------------------------

        reranker_scores.append(
            float(score)
        )

    context = "\n\n".join(
        context_parts
    )

    return {

        "context": context,

        "sources": sources,

        "reranker_scores":
            reranker_scores
    }


# ======================================================
# EVALUATE RETRIEVAL NODE
# ======================================================

def evaluate_retrieval_node(state):

    context = state.get(
        "context",
        ""
    )

    # --------------------------------------------------
    # No context
    # --------------------------------------------------

    if not context:

        return {
            "retrieval_relevant": False
        }

    # --------------------------------------------------
    # Evaluate retrieved context
    # --------------------------------------------------

    is_relevant = evaluate_retrieval(

        question=state["question"],

        context=context
    )

    return {
        "retrieval_relevant": is_relevant
    }


# ======================================================
# GENERATE ANSWER NODE
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

    history_text = ""

    for message in history[-6:]:

        history_text += (
            f'{message.get("role", "")}: '
            f'{message.get("content", "")}\n'
        )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are an AI Study Assistant.

Answer the user's question using ONLY
the provided document context.

Conversation History:

{history_text}

Document Context:

{context}

Current User Question:

{question}

Rules:

- Use the document context as the source of truth.
- Do not invent information.
- Do not use outside knowledge.
- Conversation history is only used to understand
  references and context.
- If the answer is not present in the documents,
  say you could not find the information.
- Give a clear and concise answer.

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }


# ======================================================
# RETRY NODE
# ======================================================

def retry_node(state):

    retry_count = state.get(
        "retry_count",
        0
    )

    return {
        "retry_count": retry_count + 1
    }


# ======================================================
# REJECT NODE
# ======================================================

def reject_node(state):

    return {

        "answer": (
            "I could not find that information "
            "in the uploaded documents."
        ),

        "sources": []
    }


# ======================================================
# SUMMARY NODE
# ======================================================

def summary_node(state):

    context = state.get(
        "context",
        ""
    )

    # --------------------------------------------------
    # No context
    # --------------------------------------------------

    if not context:

        return {

            "answer": (
                "I could not find enough information "
                "in the uploaded documents "
                "to create a summary."
            ),

            "sources": []
        }

    # --------------------------------------------------
    # Summary prompt
    # --------------------------------------------------

    prompt = f"""
You are an AI Study Assistant.

Create a concise summary using ONLY
the provided document context.

Rules:

- Use only the provided context.
- Do not invent information.
- Do not use outside knowledge.
- Organize the summary clearly.
- Keep important facts, technologies, dates,
  roles, and achievements when available.

Document Context:

{context}

User Request:

{state["question"]}

Summary:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }


# ======================================================
# QUIZ NODE
# ======================================================

def quiz_node(state):

    context = state.get(
        "context",
        ""
    )

    # --------------------------------------------------
    # No context
    # --------------------------------------------------

    if not context:

        return {

            "answer": (
                "I could not find enough information "
                "in the uploaded documents "
                "to generate a quiz."
            ),

            "sources": []
        }

    # --------------------------------------------------
    # Quiz prompt
    # --------------------------------------------------

    prompt = f"""
You are an AI Study Assistant.

Generate a quiz using ONLY the
provided document context.

User Request:

{state["question"]}

Document Context:

{context}

Rules:

- Use ONLY the document context.
- Do not use outside knowledge.
- Do not invent facts.
- Create clear multiple-choice questions.
- Each question must have exactly 4 options.
- Provide exactly one correct answer.
- Include the correct answer after each question.
- Keep questions directly related to the uploaded documents.

Format:

Question 1:
<question>

A. <option>
B. <option>
C. <option>
D. <option>

Correct Answer:
<option letter and answer>

Question 2:
...

Quiz:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }