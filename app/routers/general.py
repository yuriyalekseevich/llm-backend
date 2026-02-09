from fastapi import APIRouter, Depends
import logging
from app.dependencies import get_chroma_repo, get_embedder

router = APIRouter(tags=["General"])  # No prefix, so these are at root level (/, /health, /info)
logger = logging.getLogger(__name__)

# Root endpoint
@router.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {
        "project": "LLM Practicum - Week 5",
        "goal": "Embeddings & Vector Search (RAG Basics)",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "add_documents": "POST /rag/chunks",
            "search": "POST /rag/search"
        }
    }

# Health check (dynamic document count from repo)
@router.get("/health")
def health(repo=Depends(get_chroma_repo)):
    count = repo.count()
    logger.info(f"Health check: {count} documents in collection")
    return {
        "status": "healthy",
        "chromadb": "ready",
        "embeddings": "ready",
        "documents_count": count
    }

# Info endpoint (dynamic model details from embedder)
@router.get("/info")
def info(embedder=Depends(get_embedder)):
    model_path = embedder._first_module().auto_model.config._name_or_path
    dimension = embedder.get_sentence_embedding_dimension()
    logger.info(f"Info endpoint: Using model {model_path} (dim: {dimension})")
    return {
        "week": "Week 5: Embeddings & Vector Search",
        "embedding_model": model_path,
        "embedding_dimension": dimension,
        "vector_db": "ChromaDB",
        "distance_metric": "cosine"
    }