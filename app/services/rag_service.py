# app/services/rag_service.py

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

from app.models.rag import ChunkIn, ChunkOut, QueryRequest, QueryResponse
from app.repositories.chroma_repo import ChromaRepository
from app.utils.logging_config import logger

class RagService:
    def __init__(
        self,
        embedder: SentenceTransformer,
        repo: ChromaRepository
    ):
        self.embedder = embedder
        self.repo = repo
        logger.info("RagService initialized")

    def add_chunks(self, chunks: List[ChunkIn]) -> int:
        """Embed and add chunks to repo."""
        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)  # Normalize for cosine
        embeddings_list = embeddings.tolist()
        
        return self.repo.add_documents(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=texts
        )

    def search(self, req: QueryRequest) -> QueryResponse:
        """Embed query and search."""
        query_embedding = self.embedder.encode([req.query], normalize_embeddings=True).tolist()
        
        results = self.repo.query(
            query_embeddings=query_embedding,
            n_results=req.top_k
        )
        
        hits = []
        for i in range(len(results['ids'][0])):
            score = results['distances'][0][i]  # Cosine distance (0-2, but normalized ~0-1)
            # Normalize score to 0-1 (lower better)
            norm_score = score / 2.0 if score > 1 else score
            
            hit = ChunkOut(
                id=results['ids'][0][i],
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=norm_score
            )
            hits.append(hit)
        
        return QueryResponse(query=req.query, hits=hits)