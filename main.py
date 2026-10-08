import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure current directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure formal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pk_law_agent")

from graph import build_graph

# FastAPI App Setup
app = FastAPI(
    title="Pakistan Law AI Agent API",
    description="PECA 2016 Agentic Workflow Endpoint",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    category: str | None = None
    status: str
    answer: str

app_graph = build_graph()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "law": "Prevention of Electronic Crimes Act, 2016"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_agent(payload: QueryRequest):
    try:
        result = app_graph.invoke({"question": payload.question})
        return QueryResponse(
            question=payload.question,
            category=result.get("category", "N/A"),
            status=result.get("status", "completed"),
            answer=result.get("answer", "No answer generated.")
        )
    except Exception as e:
        logger.error(f"Execution error for query '{payload.question}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("=== Starting Formal Agent Evaluation Tests ===")
    
    test_suite = [
        {"category": "in_scope_answerable", "question": "What is the punishment for cyber stalking?"},
        {"category": "out_of_scope", "question": "How do I bake a chocolate cake?"},
        {"category": "in_scope_unanswerable", "question": "How many people were prosecuted under this Act in its first year after enactment?"}
    ]
    
    for test in test_suite:
        logger.info(f"Target Category: {test['category']}")
        logger.info(f"Question: {test['question']}")
        
        output = app_graph.invoke({"question": test['question']})
        
        logger.info(f"Result Status: {output.get('status')}")
        logger.info(f"Final Answer: {output.get('answer')}\n" + "-"*50)