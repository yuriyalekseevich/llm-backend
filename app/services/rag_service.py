# services/rag_service.py
from sentence_transformers import SentenceTransformer
from typing import List
from app.models.rag import ChunkIn, QueryRequest, QueryResponse, ChunkOut
from app.repositories.chroma_repo import ChromaRepository
import logging

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, embedder: SentenceTransformer, repo: ChromaRepository):
        self.embedder = embedder
        self.repo = repo
        logger.info("RagService initialized with EmbeddingGemma-300m")

    def _embed(self, text: str, is_query: bool = False) -> List[float]:
        prefix = "query: " if is_query else "passage: "
        emb = self.embedder.encode(prefix + text, normalize_embeddings=True)
        return emb.tolist()

    def add_chunks(self, chunks: List[ChunkIn]) -> int:
        if not chunks:
            return 0

        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = [self._embed(t, is_query=False) for t in texts]
        metadatas = [c.metadata for c in chunks]

        self.repo.add(ids, texts, embeddings, metadatas)
        logger.info("Added %d chunks to Chroma", len(chunks))
        return len(chunks)

    def search(self, req: QueryRequest) -> QueryResponse:
        query_emb = self._embed(req.query, is_query=True)
        raw = self.repo.query(query_emb, n_results=req.top_k)

        hits = []
        for i in range(len(raw.get("ids", [[]])[0])):
            hits.append(
                ChunkOut(
                    id=raw["ids"][0][i],
                    text=raw["documents"][0][i],
                    metadata=raw["metadatas"][0][i],
                    score=raw["distances"][0][i] if raw.get("distances") else None
                )
            )

        logger.info("RAG search completed", extra={"query": req.query, "hits": len(hits)})
        return QueryResponse(query=req.query, hits=hits)