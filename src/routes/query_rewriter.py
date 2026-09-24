from src.services.llm_service import llm


def rewrite_query(question: str) -> str:

    prompt = f"""
You are a search query optimizer for a RAG system.

Rewrite the user's question into a concise search query
that will retrieve the most relevant information from
a document.

Rules:
- Keep the original meaning.
- Remove unnecessary words.
- Include important keywords.
- Return ONLY the rewritten search query.
- Do not answer the question.

User question:
{question}

Rewritten search query:
"""

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    return rewritten_query