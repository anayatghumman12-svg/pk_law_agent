"""Runs the PECA 2016 law agent as a FastAPI app, or as a CLI test suite.

CLI:    python main.py
Server: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import logging
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import build_graph
from agent.state import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pk_law_agent")

# Build the graph once at import time; every request reuses the compiled graph.
app_graph = build_graph()

app = FastAPI(
    title="Pakistan Law AI Agent API",
    description="PECA 2016 agentic workflow endpoint",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request body for POST /ask."""

    question: str


class QueryResponse(BaseModel):
    """Response body for POST /ask, mirroring the agent's final state."""

    question: str
    in_scope: Optional[bool] = None
    scope_reason: str = ""
    status: str
    answer: str
    execution_path: List[str]


@app.get("/health")
def health_check() -> dict:
    """
    Simple status check.

    Returns:
        dict: Service status and the law the agent covers.
    """
    return {"status": "ok", "law": "Prevention of Electronic Crimes Act, 2016"}


@app.post("/ask", response_model=QueryResponse)
def ask_agent(payload: QueryRequest) -> QueryResponse:
    """
    Runs one question through the agent graph.

    Args:
        payload (QueryRequest): Contains the user's question.

    Returns:
        QueryResponse: Final answer plus status and the execution path taken.

    Raises:
        HTTPException: 500 if the graph itself fails unexpectedly.
    """
    try:
        result = app_graph.invoke(AgentState(question=payload.question))
    except Exception as e:
        logger.error(f"Execution error for query '{payload.question}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Q: {payload.question} | Path: {' -> '.join(result['execution_path'])}")
    return QueryResponse(
        question=payload.question,
        in_scope=result["in_scope"],
        scope_reason=result["scope_reason"],
        status=result["status"],
        answer=result["final_answer"],
        execution_path=result["execution_path"],
    )


def run_tests() -> None:
    """
    Runs one question per category through the graph and logs each execution path.

    Returns:
        None.
    """
    test_suite = [
        ("in_scope_answerable", "What is the punishment for cyber stalking?"),
        ("out_of_scope", "How do I bake a chocolate cake?"),
        ("in_scope_unanswerable", "How many people were prosecuted under this Act in its first year after enactment?"),
    ]
    for category, question in test_suite:
        result = app_graph.invoke(AgentState(question=question))
        logger.info(f"Category: {category}")
        logger.info(f"Q: {question}")
        logger.info(f"Path: {' -> '.join(result['execution_path'])}")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Answer: {result['final_answer'][:150]}\n" + "-" * 50)


if __name__ == "__main__":
    run_tests()