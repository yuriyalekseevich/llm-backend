from fastapi import Depends
from sentence_transformers import SentenceTransformer
from repositories.chroma_repo import ChromaRepository
from services.rag_service import RagService
import os
import logging
from openai import OpenAI
from groq import Groq

logger = logging.getLogger(__name__)



def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("openai.api_key.missing")
        raise ValueError("OPENAI_API_KEY not set in .env")
    # Do not log the key itself; only reveal length for debugging
    logger.info("openai.client.created", extra={"api_key_len": len(api_key)})
    return OpenAI(api_key=api_key)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("groq.api_key.missing")
        raise ValueError("GROQ_API_KEY not set in .env")
    logger.info("groq.client.created", extra={"api_key_len": len(api_key)})
    return Groq(api_key=api_key)


_embedder = None
_chroma_repo = None
_rag_service = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading google/embeddinggemma-300m ...")
        _embedder = SentenceTransformer("google/embeddinggemma-300m", device="mps")  # mps для Mac M1/M2/M3
        logger.info("EmbeddingGemma loaded")
    return _embedder


def get_chroma_repo() -> ChromaRepository:
    global _chroma_repo
    if _chroma_repo is None:
        _chroma_repo = ChromaRepository()
    return _chroma_repo


def get_rag_service(
    embedder=Depends(get_embedder),
    repo=Depends(get_chroma_repo)
) -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService(embedder, repo)
    return _rag_service