# ======================================================
# QUESTION / ANSWER SERVICE
# ======================================================

from src.services.agent.graph import agent_graph

from src.services.conversation_service import (
    get_history,
    save_message,
)


# ======================================================
# ANSWER QUESTION
# ======================================================

def answer_question(
    question: str,
    user_id: str,
    conversation_id: str,
):
    """
    Main orchestration service.

    Flow:

    1. Validate question
    2. Load previous conversation history
    3. Build LangGraph state
    4. Execute agent graph
    5. Save user message
    6. Save assistant response
    7. Return normalized API response
    """

    # --------------------------------------------------
    # 1. Validate question
    # --------------------------------------------------

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------
    # 2. Load previous conversation history
    #
    # IMPORTANT:
    # Do this BEFORE saving the current user message.
    # This allows follow-up questions to resolve context.
    # --------------------------------------------------

    history = get_history(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if not history:
        history = []

    # --------------------------------------------------
    # 3. Build initial LangGraph state
    # --------------------------------------------------

    initial_state = {
        # Current request
        "question": question,

        # Authenticated user
        "user_id": user_id,

        # Conversation
        "conversation_id": conversation_id,

        # Previous messages
        "history": history,

        # Intent
        "intent": "GENERAL",

        # Retrieval
        "search_queries": [],
        "rewritten_query": "",
        "context": "",
        "sources": [],
        "reranker_scores": [],
        "retrieval_relevant": False,
        "retrieval_attempted": False,

        # Retry control
        "retry_count": 0,
        "max_retries": 1,

        # Final answer
        "answer": "",
    }

    # --------------------------------------------------
    # 4. Run LangGraph
    # --------------------------------------------------

    result = agent_graph.invoke(
        initial_state
    )

    # --------------------------------------------------
    # 5. Extract final answer
    # --------------------------------------------------

    answer = str(
        result.get(
            "answer",
            ""
        )
    ).strip()

    if not answer:
        answer = (
            "I was unable to generate a response."
        )

    # --------------------------------------------------
    # 6. Save user message
    # --------------------------------------------------

    save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=question,
    )

    # --------------------------------------------------
    # 7. Save assistant message
    # --------------------------------------------------

    save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=answer,
    )

    # --------------------------------------------------
    # 8. Return API response
    # --------------------------------------------------

    return {
        "question": question,

        "intent": result.get(
            "intent",
            "GENERAL"
        ),

        "rewritten_query": result.get(
            "rewritten_query"
        ),

        "retrieval_relevant": result.get(
            "retrieval_relevant"
        ),

        "answer": answer,

        "sources": result.get(
            "sources",
            []
        ),

        "reranker_scores": result.get(
            "reranker_scores",
            []
        ),
    }
