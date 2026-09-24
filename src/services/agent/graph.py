from langgraph.graph import (
    StateGraph,
    START,
    END
)

from src.services.agent.state import AgentState

from src.services.agent.nodes import (
    classify_intent_node,
    rewrite_query_node,
    retrieve_documents_node,
    evaluate_retrieval_node,
    generate_answer_node,
    summary_node,
    quiz_node,
    retry_node,
    reject_node
)


MAX_RETRIES = 1


# ======================================================
# ROUTE AFTER INTENT
# ======================================================

def route_after_intent(
    state: AgentState
):

    # All intents need document retrieval first.
    return "rewrite_query"


# ======================================================
# ROUTE AFTER EVALUATION
# ======================================================

def route_after_evaluation(
    state: AgentState
):

    retrieval_relevant = state.get(
        "retrieval_relevant",
        False
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    intent = state.get(
        "intent",
        "QA"
    )

    # --------------------------------------------------
    # Good retrieval
    # --------------------------------------------------

    if retrieval_relevant:

        if intent == "SUMMARY":
            return "summary"

        if intent == "QUIZ":
            return "quiz"

        return "generate"

    # --------------------------------------------------
    # Retry
    # --------------------------------------------------

    if retry_count < MAX_RETRIES:

        return "retry"

    # --------------------------------------------------
    # Reject
    # --------------------------------------------------

    return "reject"


# ======================================================
# BUILD GRAPH
# ======================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    # ==================================================
    # NODES
    # ==================================================

    graph.add_node(
        "classify_intent",
        classify_intent_node
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query_node
    )

    graph.add_node(
        "retrieve",
        retrieve_documents_node
    )

    graph.add_node(
        "evaluate",
        evaluate_retrieval_node
    )

    graph.add_node(
        "generate",
        generate_answer_node
    )

    graph.add_node(
        "summary",
        summary_node
    )

    graph.add_node(
        "quiz",
        quiz_node
    )

    graph.add_node(
        "retry",
        retry_node
    )

    graph.add_node(
        "reject",
        reject_node
    )

    # ==================================================
    # START
    # ==================================================

    graph.add_edge(
        START,
        "classify_intent"
    )

    # ==================================================
    # INTENT → REWRITE
    # ==================================================

    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "rewrite_query": "rewrite_query"
        }
    )

    # ==================================================
    # REWRITE → RETRIEVE
    # ==================================================

    graph.add_edge(
        "rewrite_query",
        "retrieve"
    )

    # ==================================================
    # RETRIEVE → EVALUATE
    # ==================================================

    graph.add_edge(
        "retrieve",
        "evaluate"
    )

    # ==================================================
    # EVALUATE → ROUTER
    # ==================================================

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "summary": "summary",
            "quiz": "quiz",
            "retry": "retry",
            "reject": "reject"
        }
    )

    # ==================================================
    # RETRY → REWRITE
    # ==================================================

    graph.add_edge(
        "retry",
        "rewrite_query"
    )

    # ==================================================
    # END NODES
    # ==================================================

    graph.add_edge(
        "generate",
        END
    )

    graph.add_edge(
        "summary",
        END
    )

    graph.add_edge(
        "quiz",
        END
    )

    graph.add_edge(
        "reject",
        END
    )

    # ==================================================
    # COMPILE
    # ==================================================

    return graph.compile()


agent_graph = build_graph()