"""The agent's three runtime tools: scope_checker, retriever, answer_generator."""
import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from agent.state import AgentState, RetrievedChunk

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Connect once, at import time, to the vector store built by indexing/build_index.py.
# Tools never rebuild or re-embed the PDF - that already happened offline.
_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
_vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=_embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.60  # below this, we treat the retrieval as not useful


def retriever(state: AgentState) -> AgentState:
    """
    Searches the vector store for chunks relevant to the question.

    Runs only when the Scope Checker has judged the question in-scope.
    Populates retrieved_chunks and retrieval_sufficient on the state.

    Args:
        state (AgentState): Current agent state (must have `question` set).

    Returns:
        AgentState: Updated state with retrieved_chunks and retrieval_sufficient filled in.
    """
    state.execution_path.append("retriever")

    try:
        results = _vectorstore.similarity_search_with_score(state.question, k=RETRIEVAL_TOP_K)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state.retrieval_sufficient = False
        return state

    chunks = []
    for document, distance in results:
        similarity = round(1.0 - float(distance), 4)
        chunks.append(
            RetrievedChunk(
                text=document.page_content,
                section=str(document.metadata.get("section", "unknown")),
                score=similarity
            )
        )

    state.retrieved_chunks = chunks

    # Retrieval is only "sufficient" if we have results AND the best one clears the
    # relevance bar - a non-empty list of irrelevant chunks should still lead to Decline.
    top_score = chunks[0].score if chunks else 0.0
    state.retrieval_sufficient = top_score >= RETRIEVAL_MIN_SCORE

    logger.info(f"Retrieved {len(chunks)} chunks, top score {top_score:.3f}")
    return state