from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.services.agent.state import (
    AgentState,
)

from src.services.agent.nodes import (
    classify_intent_node,
    rewrite_query_node,
    retrieve_documents_node,
    evaluate_retrieval_node,
    generate_answer_node,
    general_answer_node,
    coding_answer_node,
    summary_node,
    quiz_node,
    retry_node,
    reject_node,
)


# ======================================================
# INTENT ROUTING
# ======================================================

def route_after_intent(
    state: AgentState
):

    intent =
        state.get(
            "intent",
            "GENERAL"
        )


    # ----------------------------------------------
    # RAG
    # ----------------------------------------------

    if intent in {
        "QA",
        "SUMMARY",
        "QUIZ",
    }:

        return "rag"


    # ----------------------------------------------
    # CODING
    # ----------------------------------------------

    if intent == "CODING":

        return "coding"


    # ----------------------------------------------
    # GENERAL
    # ----------------------------------------------

    return "general"


# ======================================================
# RETRIEVAL ROUTING
# ======================================================

def route_after_evaluation(
    state: AgentState
):

    relevant =
        bool(
            state.get(
                "retrieval_relevant",
                False
            )
        )


    if relevant:

        intent =
            state.get(
                "intent",
                "QA"
            )


        if intent == "SUMMARY":
            return "summary"


        if intent == "QUIZ":
            return "quiz"


        return "qa"


    # ----------------------------------------------
    # Retry
    # ----------------------------------------------

    retry_count =
        int(
            state.get(
                "retry_count",
                0
            )
        )


    max_retries =
        int(
            state.get(
                "max_retries",
                1
            )
        )


    if retry_count < max_retries:

        return "retry"


    return "reject"


# ======================================================
# GRAPH
# ======================================================

builder =
    StateGraph(
        AgentState
    )


# ======================================================
# NODES
# ======================================================

builder.add_node(
    "classify_intent",
    classify_intent_node,
)

builder.add_node(
    "rewrite_query",
    rewrite_query_node,
)

builder.add_node(
    "retrieve",
    retrieve_documents_node,
)

builder.add_node(
    "evaluate",
    evaluate_retrieval_node,
)

builder.add_node(
    "qa",
    generate_answer_node,
)

builder.add_node(
    "summary",
    summary_node,
)

builder.add_node(
    "quiz",
    quiz_node,
)

builder.add_node(
    "coding",
    coding_answer_node,
)

builder.add_node(
    "general",
    general_answer_node,
)

builder.add_node(
    "retry",
    retry_node,
)

builder.add_node(
    "reject",
    reject_node,
)


# ======================================================
# START
# ======================================================

builder.add_edge(
    START,
    "classify_intent",
)


# ======================================================
# ROUTE BY INTENT
# ======================================================

builder.add_conditional_edges(

    "classify_intent",

    route_after_intent,

    {

        "rag":
            "rewrite_query",

        "coding":
            "coding",

        "general":
            "general",
    },
)


# ======================================================
# RAG PIPELINE
# ======================================================

builder.add_edge(
    "rewrite_query",
    "retrieve",
)

builder.add_edge(
    "retrieve",
    "evaluate",
)


# ======================================================
# RETRIEVAL DECISION
# ======================================================

builder.add_conditional_edges(

    "evaluate",

    route_after_evaluation,

    {

        "qa":
            "qa",

        "summary":
            "summary",

        "quiz":
            "quiz",

        "retry":
            "retry",

        "reject":
            "reject",
    },
)


# ======================================================
# RETRY
# ======================================================

builder.add_edge(
    "retry",
    "rewrite_query",
)


# ======================================================
# END
# ======================================================

builder.add_edge(
    "qa",
    END,
)

builder.add_edge(
    "summary",
    END,
)

builder.add_edge(
    "quiz",
    END,
)

builder.add_edge(
    "coding",
    END,
)

builder.add_edge(
    "general",
    END,
)

builder.add_edge(
    "reject",
    END,
)


# ======================================================
# COMPILE
# ======================================================

agent_graph =
    builder.compile()
