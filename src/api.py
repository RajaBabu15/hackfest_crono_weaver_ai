# python src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from src.rag import get_chat_engine

logger = logging.getLogger(__name__)

app = FastAPI(title="Realtime RAG API")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question")

class SourceNodeModel(BaseModel):
    id: str
    text: Optional[str] = None
    metadata: Optional[dict] = {}
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[SourceNodeModel] = Field(default_factory=list, description="List of source documents used")

@app.on_event("startup")
async def startup_event():
    try:
        get_chat_engine()
        logger.info("API started successfully, chat engine loaded.")
    except Exception as e:
        logger.error(f"API startup failed to load chat engine: {e}", exc_info=True)
        # Depending on severity, you might want to prevent startup

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        chat_engine = get_chat_engine()
        logger.info(f"Received query: {request.query}")
        response = await chat_engine.achat(request.query)

        sources_data = []
        if hasattr(response, 'source_nodes'):
             for node in response.source_nodes:
                 sources_data.append(SourceNodeModel(
                     id=node.node_id,
                     text=node.get_content(metadata_mode="all"), # Or adjust as needed
                     metadata=node.metadata or {},
                     score=node.score
                 ))

        return QueryResponse(answer=str(response.response), sources=sources_data)
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")