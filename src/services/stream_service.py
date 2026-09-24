import json

from src.services.agent.graph import agent_graph

from src.services.conversation_service import (
    get_history,
    save_message
)


async def stream_chat(
    question: str,
    user_id: str,
    conversation_id: str
):

    # ==================================================
    # LOAD MEMORY
    # ==================================================

    history = get_history(
        conversation_id=conversation_id,
        user_id=user_id
    )

    # ==================================================
    # INITIAL STATE
    # ==================================================

    initial_state = {

        "question": question,

        "user_id": user_id,

        "conversation_id": conversation_id,

        "history": history,

        "rewritten_query": "",

        "context": "",

        "sources": [],

        "retrieval_relevant": False,

        "answer": "",

        "retry_count": 0
    }

    # ==================================================
    # VARIABLES
    # ==================================================

    answer_parts = []

    final_state = {}

    # ==================================================
    # RUN LANGGRAPH
    # ==================================================

    async for chunk in agent_graph.astream(
        initial_state,

        stream_mode=[
            "messages",
            "values"
        ],

        version="v2"
    ):

        # ==============================================
        # LLM MESSAGE STREAM
        # ==============================================

        if chunk["type"] == "messages":

            message_chunk, metadata = chunk["data"]

            node_name = metadata.get(
                "langgraph_node"
            )

            # Only final generation
            if node_name != "generate":
                continue

            content = message_chunk.content

            if not content:
                continue

            answer_parts.append(
                content
            )

            yield (
                f"data: {json.dumps({
                    'type': 'token',
                    'content': content
                })}\n\n"
            )

        # ==============================================
        # GRAPH STATE
        # ==============================================

        elif chunk["type"] == "values":

            final_state = chunk["data"]

    # ==================================================
    # FINAL ANSWER
    # ==================================================

    final_answer = final_state.get(
        "answer",
        "".join(answer_parts)
    )

    rewritten_query = final_state.get(
        "rewritten_query",
        ""
    )

    retrieval_relevant = final_state.get(
        "retrieval_relevant",
        False
    )

    sources = final_state.get(
        "sources",
        []
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
        content=final_answer
    )

    # ==================================================
    # FINAL EVENT
    # ==================================================

    yield (
        f"data: {json.dumps({
            'type': 'complete',
            'conversation_id': conversation_id,
            'rewritten_query': rewritten_query,
            'retrieval_relevant': retrieval_relevant,
            'sources': sources
        })}\n\n"
    )