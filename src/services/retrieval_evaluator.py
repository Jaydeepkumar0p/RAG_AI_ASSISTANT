from src.services.llm_service import llm


def evaluate_retrieval(
    question: str,
    context: str
) -> bool:

    prompt = f"""
You are a retrieval evaluator for a RAG system.

Determine whether the provided context contains
information that can help answer the user's question.

Return ONLY one word:

YES
or
NO

User Question:
{question}

Retrieved Context:
{context}

Is the context relevant?
"""

    response = llm.invoke(prompt)

    result = response.content.strip().upper()

    return result.startswith("YES")