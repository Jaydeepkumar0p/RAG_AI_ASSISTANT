from typing import TypedDict


class AgentState(TypedDict, total=False):
    # Request
    question: str
    user_id: str
    conversation_id: str
    history: list[dict]

    # Intent
    intent: str

    # Retrieval
    search_queries: list[str]
    rewritten_query: str
    context: str
    sources: list[dict]
    reranker_scores: list[float]
    retrieval_relevant: bool
    retrieval_attempted: bool

    # Retry
    retry_count: int
    max_retries: int

    # Final response
    answer: str
