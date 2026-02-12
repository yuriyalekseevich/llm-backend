from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import tempfile
import os
import shutil
import logging

from app.models.rag import (
    ChunkIn,
    QueryRequest,
    QueryResponse,
    IngestionResponse,
    StatsResponse
)
from app.services.rag_service import RagService
from app.dependencies import get_rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = logging.getLogger(__name__)


@router.post("/chunks", response_model=int)
async def add_chunks(
    chunks: List[ChunkIn],
    service: RagService = Depends(get_rag_service)
):
    """
    Ingest text chunks directly (replaces /ingest/text)
    Returns the number of chunks actually added.
    """
    try:
        count = service.add_chunks(chunks)
        logger.info(f"Added {count} chunks via API")
        return count
    except Exception as e:
        logger.exception("Failed to add chunks")
        raise HTTPException(500, detail=str(e))


@router.post("/ingest/pdf", response_model=IngestionResponse)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    service: RagService = Depends(get_rag_service)
):
    """
    Upload and asynchronously ingest a PDF file.
    Processing happens in background → immediate response.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Only PDF files are supported")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        background_tasks.add_task(
            _process_pdf_background,
            tmp_path,
            file.filename,
            chunk_size,
            chunk_overlap,
            service
        )

        return IngestionResponse(
            status="processing",
            chunks_added=0,
            message=f"PDF ingestion queued: {file.filename}"
        )

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.exception("PDF ingestion setup failed")
        raise HTTPException(500, detail=str(e))


async def _process_pdf_background(
    pdf_path: str,
    filename: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    service: RagService
):
    """Background PDF → chunks → vector store task"""
    logger.info(f"Starting background PDF ingestion: {filename}")

    try:
        docs = service.load_pdf(pdf_path)
        chunks = service.chunk_docs(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        added = service.ingest_chunks(chunks, source=filename)

        logger.info(
            f"PDF ingestion completed",
            extra={"filename": filename, "chunks_added": added}
        )

    except Exception as e:
        logger.exception(
            "Background PDF ingestion failed",
            extra={"filename": filename}
        )
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


@router.post("/search", response_model=QueryResponse)
async def search(
    request: QueryRequest,
    service: RagService = Depends(get_rag_service)
):
    """
    Semantic / vector search — returns retrieved chunks + scores
    """
    try:
        result = service.search(request)
        logger.info(f"Search completed — query: {request.query[:80]}…")
        return result
    except Exception as e:
        logger.exception("Search endpoint failed")
        raise HTTPException(500, detail=str(e))


@router.post("/query", response_model=Dict[str, Any])
async def query(
    request: QueryRequest,
    service: RagService = Depends(get_rag_service)
):
    """
    Full RAG pipeline: retrieve → LLM generate answer + citations
    """
    try:
        result = service.rag_query(
            query=request.query,
            top_k=request.top_k
        )
        logger.info(f"RAG query completed — query: {request.query[:80]}…")
        return result
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    service: RagService = Depends(get_rag_service)
):
    """Return vector store / collection statistics"""
    try:
        stats = service.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.exception("Failed to retrieve stats")
        raise HTTPException(500, detail=str(e))


@router.delete("/reset")
async def reset(
    service: RagService = Depends(get_rag_service)
):
    """
    ⚠️ Delete ALL documents in the collection / reset vector store
    """
    logger.warning("Collection reset requested")
    try:
        success = service.repo.delete_collection()
        if success:
            return {"status": "success", "message": "Collection has been reset"}
        else:
            raise RuntimeError("Delete collection returned false")
    except Exception as e:
        logger.exception("Reset failed")
        raise HTTPException(500, detail=str(e))