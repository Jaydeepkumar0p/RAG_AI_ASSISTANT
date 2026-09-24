from typing import TypedDict, List


class AgentState(TypedDict):

    # ==================================================
    # USER
    # ==================================================

    question: str
    user_id: str

    # ==================================================
    # CONVERSATION
    # ==================================================

    conversation_id: str
    history: List[dict]

    # ==================================================
    # INTENT
    # ==================================================

    intent: str

    # ==================================================
    # RETRIEVAL
    # ==================================================

    search_queries: List[str]

    rewritten_query: str

    context: str

    sources: List[dict]

    reranker_scores: List[float]

    retrieval_relevant: bool

    # ==================================================
    # ANSWER
    # ==================================================

    answer: str

    # ==================================================
    # RETRY
    # ==================================================

    retry_count: int