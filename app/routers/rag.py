from fastapi import APIRouter, Depends, HTTPException
from app.models.rag import ChunkIn, QueryRequest, QueryResponse
from app.services.rag_service import RagService
from app.dependencies import get_rag_service
from typing import List
import logging

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = logging.getLogger(__name__)

# Comment: Endpoint to add chunks.
# Returns count added; raises HTTP errors on failure.
@router.post("/chunks", response_model=int)
def add_chunks(
    chunks: List[ChunkIn],
    service: RagService = Depends(get_rag_service)
):
    try:
        count = service.add_chunks(chunks)
        logger.info(f"Added {count} chunks via API")
        return count
    except Exception as e:
        logger.exception("Error in add_chunks endpoint")
        raise HTTPException(500, str(e))

# Comment: Endpoint for search.
# Returns formatted response; handles errors.
@router.post("/search", response_model=QueryResponse)
def search(
    req: QueryRequest,
    service: RagService = Depends(get_rag_service)
):
    try:
        response = service.search(req)
        logger.info(f"Search endpoint completed for query: '{req.query}'")
        return response
    except Exception as e:
        logger.exception("Error in search endpoint")
        raise HTTPException(500, str(e))