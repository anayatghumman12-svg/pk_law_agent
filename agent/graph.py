"""Wires the three tools and the decline node into a LangGraph StateGraph."""
import logging

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.tools import scope_checker, retriever, answer_generator, decline

logger = logging.getLogger(__name__)


def _route_after_scope_check(state: AgentState) -> str:
    """
    Decides where to go after the Scope Checker runs.

    Args:
        state (AgentState): Current state (in_scope must be set).

    Returns:
        str: "retriever" if in scope, otherwise "decline".
    """
    return "retriever" if state.in_scope else "decline"


def _route_after_retrieval(state: AgentState) -> str:
    """
    Decides where to go after the Retriever runs.

    Args:
        state (AgentState): Current state (retrieval_sufficient must be set).

    Returns:
        str: "answer_generator" if retrieval found something useful, otherwise "decline".
    """
    return "answer_generator" if state.retrieval_sufficient else "decline"


def build_graph():
    """
    Builds and compiles the agent's LangGraph workflow.

    Workflow: scope_checker -> (retriever | decline)
              retriever -> (answer_generator | decline)
              answer_generator -> END
              decline -> END

    Returns:
        A compiled LangGraph graph, ready to .invoke() with an AgentState.
    """
    graph = StateGraph(AgentState)

    graph.add_node("scope_checker", scope_checker)
    graph.add_node("retriever", retriever)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("decline", decline)

    graph.set_entry_point("scope_checker")

    graph.add_conditional_edges(
        "scope_checker",
        _route_after_scope_check,
        {"retriever": "retriever", "decline": "decline"}
    )
    graph.add_conditional_edges(
        "retriever",
        _route_after_retrieval,
        {"answer_generator": "answer_generator", "decline": "decline"}
    )

    graph.add_edge("answer_generator", END)
    graph.add_edge("decline", END)

    return graph.compile()