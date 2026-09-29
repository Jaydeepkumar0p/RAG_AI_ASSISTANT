# ======================================================
# QUESTION / ANSWER SERVICE
# ======================================================

from src.services.agent.graph import (
    agent_graph,
)

from src.services.conversation_service import (
    get_history,
    get_conversation,
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

    question =
        question.strip()


    if not question:

        raise ValueError(
            "Question cannot be empty."
        )


    # --------------------------------------------------
    # Validate conversation
    # --------------------------------------------------

    conversation =
        get_conversation(

            conversation_id=
                conversation_id,

            user_id=
                user_id,
        )


    if not conversation:

        raise ValueError(
            "Conversation not found."
        )


    # --------------------------------------------------
    # Load previous history
    #
    # IMPORTANT:
    # The current message is NOT saved yet.
    # --------------------------------------------------

    history =
        get_history(

            conversation_id=
                conversation_id,

            user_id=
                user_id,
        )


    # --------------------------------------------------
    # Initial graph state
    # --------------------------------------------------

    initial_state = {

        "question":
            question,

        "user_id":
            user_id,

        "conversation_id":
            conversation_id,

        "history":
            history,

        "intent":
            "GENERAL",

        "search_queries":
            [],

        "rewritten_query":
            "",

        "context":
            "",

        "sources":
            [],

        "reranker_scores":
            [],

        "retrieval_relevant":
            False,

        "retrieval_attempted":
            False,

        "retry_count":
            0,

        "max_retries":
            1,

        "answer":
            "",
    }


    # --------------------------------------------------
    # Run agent
    # --------------------------------------------------

    result =
        agent_graph.invoke(
            initial_state
        )


    answer =
        str(
            result.get(
                "answer",
                ""
            )
        ).strip()


    if not answer:

        answer =
            "I was unable to generate a response."


    # --------------------------------------------------
    # Save user message
    # --------------------------------------------------

    save_message(

        conversation_id=
            conversation_id,

        user_id=
            user_id,

        role=
            "user",

        content=
            question,
    )


    # --------------------------------------------------
    # Save assistant message
    # --------------------------------------------------

    save_message(

        conversation_id=
            conversation_id,

        user_id=
            user_id,

        role=
            "assistant",

        content=
            answer,
    )


    # --------------------------------------------------
    # Return API result
    # --------------------------------------------------

    return {

        "question":
            question,

        "intent":
            result.get(
                "intent",
                "GENERAL"
            ),

        "rewritten_query":
            result.get(
                "rewritten_query"
            ),

        "retrieval_relevant":
            result.get(
                "retrieval_relevant"
            ),

        "answer":
            answer,

        "sources":
            result.get(
                "sources",
                []
            ),

        "reranker_scores":
            result.get(
                "reranker_scores",
                []
            ),
    }
