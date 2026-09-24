from src.services.agent.graph import agent_graph

from src.services.conversation_service import (
    get_history,
    save_message
)


def answer_question(
    question: str,
    user_id: str,
    conversation_id: str
):

    # ==================================================
    # LOAD PREVIOUS CONVERSATION
    # ==================================================

    history = get_history(
        conversation_id=conversation_id,
        user_id=user_id
    )

    # ==================================================
    # INITIAL GRAPH STATE
    # ==================================================

    initial_state = {

        "question": question,

        "user_id": user_id,

        "conversation_id": conversation_id,

        "history": history,

        "intent": "",

        "rewritten_query": "",

        "context": "",

        "sources": [],

        "retrieval_relevant": False,

        "answer": "",

        "retry_count": 0
    }

    # ==================================================
    # RUN LANGGRAPH
    # ==================================================

    result = agent_graph.invoke(
        initial_state
    )

    # ==================================================
    # SAVE USER MESSAGE
    # ==================================================

    save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=question
    )

    # ==================================================
    # SAVE ASSISTANT MESSAGE
    # ==================================================

    save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=result.get(
            "answer",
            ""
        )
    )

    # ==================================================
    # RESPONSE
    # ==================================================

    return {

        "question": result.get(
            "question",
            question
        ),

        "intent": result.get(
            "intent",
            "QA"
        ),

        "rewritten_query": result.get(
            "rewritten_query",
            ""
        ),

        "retrieval_relevant": result.get(
            "retrieval_relevant",
            False
        ),

        "answer": result.get(
            "answer",
            ""
        ),

        "sources": result.get(
            "sources",
            []
        )
    }