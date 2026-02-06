import os
from fastapi import Depends
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from groq import Groq

from app.repositories.chroma_repo import ChromaRepository
from app.services.rag_service import RagService
from app.utils.logging_config import logger

# --- Singletons ---
_embedder: SentenceTransformer | None = None
_chroma_repo: ChromaRepository | None = None
_rag_service: RagService | None = None


# Dependency for embedder
def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        try:
            _embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")  # Adjust device: 'mps' for Mac, 'cuda' for GPU
            logger.info(f"✅ Embedding model loaded (dimension: {_embedder.get_sentence_embedding_dimension()})")
        except Exception as e:
            logger.error(f"❌ Embedding model failed: {e}")
            raise
        logger.info("Embedder loaded successfully")
    return _embedder

# Dependency for Chroma repo
def get_chroma_repo() -> ChromaRepository:
    global _chroma_repo
    if _chroma_repo is None:
        logger.info("Initializing ChromaRepository")
        _chroma_repo = ChromaRepository()  # This triggers chroma_repo.py init
    return _chroma_repo

# Dependency for RAG service (injects embedder and repo)
def get_rag_service(
    embedder: SentenceTransformer = Depends(get_embedder),
    repo: ChromaRepository = Depends(get_chroma_repo)
) -> RagService:
    global _rag_service
    if _rag_service is None:
        logger.info("Initializing RagService")
        _rag_service = RagService(embedder, repo)
    return _rag_service


# --- OpenAI Client ---
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("openai.api_key.missing")
        raise ValueError("OPENAI_API_KEY not set in environment")
    logger.info("OpenAI client created", extra={"api_key_len": len(api_key)})
    return OpenAI(api_key=api_key)


# --- Groq Client ---
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("groq.api_key.missing")
        raise ValueError("GROQ_API_KEY not set in environment")
    logger.info("Groq client created", extra={"api_key_len": len(api_key)})
    return Groq(api_key=api_key)

