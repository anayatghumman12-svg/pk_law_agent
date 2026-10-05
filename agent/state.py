"""Shared state object that flows through every node of the LangGraph agent."""
from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One chunk returned by the retriever, with its similarity score.

    Attributes:
        text: The chunk's raw text content.
        section: Section number/label the chunk belongs to.
        score: Similarity score between the query and this chunk.
    """
    text: str
    section: str
    score: float


class AgentState(BaseModel):
    """The single object every graph node reads from and writes back to.

    Attributes:
        question: The user's original question.
        in_scope: Whether the Scope Checker judged the question to be about PECA 2016.
        scope_reason: Short explanation from the Scope Checker for its decision.
        retrieved_chunks: Chunks returned by the Retriever, each with source + score.
        retrieval_sufficient: Whether the Retriever found anything useful enough to answer from.
        final_answer: The answer text shown to the user (or a decline message).
        status: Outcome of the run: "answered", "out_of_scope", or "no_results".
        execution_path: Names of nodes that ran, in order, for logging/debugging.
    """
    question: str
    in_scope: Optional[bool] = None
    scope_reason: str = ""
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_sufficient: Optional[bool] = None
    final_answer: str = ""
    status: str = "pending"
    execution_path: List[str] = Field(default_factory=list)