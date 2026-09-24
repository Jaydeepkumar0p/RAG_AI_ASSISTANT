from src.services.llm_service import llm


def generate_search_queries(
    question: str,
    history: list | None = None
):

    history = history or []

    history_text = ""

    for message in history[-6:]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        history_text += (
            f"{role}: {content}\n"
        )

    prompt = f"""
You are a search-query generation system for a RAG application.

Generate exactly 3 different search queries that could
retrieve relevant information for the user's question.

Use the conversation history to understand references
such as:

- it
- this
- that
- they
- previous answer

Rules:

- Preserve the original meaning.
- Each query should use different wording.
- Make each query useful for vector retrieval.
- Do not answer the question.
- Return ONLY the 3 queries.
- Put one query on each line.
- Do not number them.

Conversation History:

{history_text}

User Question:

{question}

Search Queries:
"""

    response = llm.invoke(
        prompt
    )

    lines = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    queries = []

    for line in lines:

        line = line.lstrip(
            "0123456789.-) "
        )

        if line:
            queries.append(line)

    # Always keep original question as fallback
    if not queries:

        queries = [question]

    # Maximum 3
    queries = queries[:3]

    return queries