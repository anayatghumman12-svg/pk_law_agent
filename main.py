"""Runs the agent against a set of test questions and prints each execution path.

Demonstrates that in-scope, out-of-scope, and in-scope-but-unanswerable
questions each take a different route through the graph.
"""
import logging

from agent.graph import build_graph
from agent.state import AgentState

logging.basicConfig(level=logging.WARNING)  # quiet the HTTP/AFC noise for this report

TEST_QUESTIONS = [
    ("What is the punishment for cyber stalking?", "in_scope_answerable"),
    ("How do I bake a chocolate cake?", "out_of_scope"),
    ("How many people were prosecuted under this Act in its first year after enactment?", "in_scope_unanswerable"),
]


def run_tests() -> None:
    """
    Runs each test question through the compiled agent graph and prints the
    question, category, resulting execution path, status, and answer.

    Returns:
        None.
    """
    app = build_graph()

    for question, category in TEST_QUESTIONS:
        result = app.invoke(AgentState(question=question))
        print(f"\nQ: {question}")
        print(f"Category: {category}")
        print(f"Path: {' -> '.join(result['execution_path'])}")
        print(f"Status: {result['status']}")
        print(f"Answer: {result['final_answer'][:150]}")


if __name__ == "__main__":
    run_tests()