from sentence_transformers import SentenceTransformer
from typing import List
from app.models.rag import ChunkIn, QueryRequest, QueryResponse, ChunkOut
from app.repositories.chroma_repo import ChromaRepository
import logging

logger = logging.getLogger(__name__)

# Comment: Service class for RAG operations.
# Embeds text and interacts with repo.
class RagService:
    def __init__(self, embedder: SentenceTransformer, repo: ChromaRepository):
        self.embedder = embedder
        self.repo = repo
        logger.info("RagService initialized with embedder and repo")

    # Comment: Internal embedding function.
    # Uses asymmetric prefixes ("query:" / "passage:") for better semantic search.
    # Normalizes embeddings for cosine similarity.
    def _embed(self, text: str, is_query: bool = False) -> List[float]:
        prefix = "query: " if is_query else "passage: "
        logger.debug(f"Embedding text with prefix '{prefix}'")
        emb = self.embedder.encode(prefix + text, normalize_embeddings=True)
        return emb.tolist()

    # Comment: Add chunks to the vector DB.
    # Returns count added; logs for visibility.
    def add_chunks(self, chunks: List[ChunkIn]) -> int:
        if not chunks:
            logger.warning("No chunks provided to add")
            return 0

        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = [self._embed(t, is_query=False) for t in texts]
        metadatas = [c.metadata for c in chunks]

        logger.info(f"Embedding and adding {len(chunks)} chunks")
        self.repo.add(ids, texts, embeddings, metadatas)
        return len(chunks)

    # Comment: Perform semantic search.
    # Embeds query, queries repo, formats results into ChunkOut.
    def search(self, req: QueryRequest) -> QueryResponse:
        logger.info(f"Starting search for query: '{req.query}' with top_k={req.top_k}")
        query_emb = self._embed(req.query, is_query=True)
        raw = self.repo.query(query_emb, n_results=req.top_k)

        hits = []
        for i in range(len(raw.get("ids", [[]])[0])):
            hit = ChunkOut(
                id=raw["ids"][0][i],
                text=raw["documents"][0][i],
                metadata=raw["metadatas"][0][i],
                score=raw["distances"][0][i] if raw.get("distances") else None
            )
            hits.append(hit)
            logger.debug(f"Hit {i+1}: ID={hit.id}, Score={hit.score}")

        logger.info(f"Search completed with {len(hits)} hits")
        return QueryResponse(query=req.query, hits=hits)