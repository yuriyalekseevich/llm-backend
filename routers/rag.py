# routers/rag.py
from fastapi import APIRouter, Depends, HTTPException
from models.rag import ChunkIn, QueryRag, RagResponse
from services.rag_service import RagService
from dependencies import get_rag_service
from typing import List
import logging

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = logging.getLogger(__name__)


@router.post("/chunks", response_model=int)
def add_chunks(
    chunks: List[ChunkIn],
    service: RagService = Depends(get_rag_service)
):
    try:
        return service.add_chunks(chunks)
    except Exception as e:
        logger.exception("rag.add_chunks.error")
        raise HTTPException(500, str(e))


@router.post("/search", response_model=RagResponse)
def search(
    req: QueryRag,
    service: RagService = Depends(get_rag_service)
):
    try:
        return service.search(req)
    except Exception as e:
        logger.exception("rag.search.error")
        raise HTTPException(500, str(e))