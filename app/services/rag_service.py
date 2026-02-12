# app/services/rag_service.py

from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import traceback

from sentence_transformers import SentenceTransformer
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks.base import BaseCallbackHandler

from langchain_groq import ChatGroq

from app.models.rag import ChunkIn, ChunkOut, QueryRequest, QueryResponse
from app.repositories.chroma_repo import ChromaRepository
from app.config import settings
from app.utils.logging_config import logger


class SentenceTransformerEmbeddingsWrapper(Embeddings):
    """LangChain Embeddings adapter for a pre-loaded SentenceTransformer model."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


class StreamingCallbackHandler(BaseCallbackHandler):
    """Collect tokens for streaming LLM responses."""

    def __init__(self):
        self.tokens: List[str] = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def get_full_response(self) -> str:
        return "".join(self.tokens)


class RagService:
    """RAG service with PDF ingestion, chunking, embedding, retrieval, and generation."""

    def __init__(self, embedder: SentenceTransformer, repo: ChromaRepository):
        self.embedder_model = embedder
        self.repo = repo

        # LangChain wrapper for embeddings (uses existing model, no duplicate load)
        self.embedder = SentenceTransformerEmbeddingsWrapper(embedder)

        # Initialize vector store (Chroma)
        self.vectorstore = Chroma(
            client=repo.client,
            collection_name=repo.collection_name,
            embedding_function=self.embedder
        )

        # LLM is lazy-initialized so vector-only use (ingest/search) works without GROQ_API_KEY
        self._llm: Optional[Any] = None

        logger.info(
            "✅ RagService initialized",
            extra={
                "llm_model": "llama-3.3-70b-versatile",
                "embedding_dim": embedder.get_sentence_embedding_dimension(),
                "collection": repo.collection_name
            }
        )

    # -------------------------------
    # PDF Loading & Chunking
    # -------------------------------

    def load_pdf(self, pdf_path: str) -> List[Any]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        loader = PyPDFLoader(str(path))
        docs = loader.load()
        logger.info(f"📄 Loaded PDF {pdf_path}, pages: {len(docs)}")
        return docs

    def chunk_docs(
        self,
        docs: List[Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Any]:
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"✂️ Chunked documents into {len(chunks)} chunks")
        return chunks

    # -------------------------------
    # Chunk Embedding & Storage
    # -------------------------------

    def add_chunks(self, chunks: List[ChunkIn]) -> int:
        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        embeddings = self.embedder_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

        result = self.repo.add_documents(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        logger.info(f"📚 Added {result} chunks to vector store")
        return result

    def ingest_chunks(self, chunks: List[Any], source: Optional[str] = None) -> int:
        chunk_in_list: List[ChunkIn] = []
        timestamp = time.time()

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source or 'doc'}_{timestamp}_{i}"
            metadata = getattr(chunk, 'metadata', {}).copy() if hasattr(chunk, 'metadata') else {}
            metadata.update({
                "source": source or metadata.get("source", "unknown"),
                "chunk_index": i,
                "ingested_at": timestamp
            })
            text = getattr(chunk, 'page_content', str(chunk))
            chunk_in_list.append(ChunkIn(id=chunk_id, text=text, metadata=metadata))

        return self.add_chunks(chunk_in_list)

    # -------------------------------
    # Semantic Search
    # -------------------------------

    def search(self, req: QueryRequest) -> QueryResponse:
        query_embedding = self.embedder_model.encode([req.query], normalize_embeddings=True).tolist()
        results = self.repo.query(query_embeddings=query_embedding, n_results=req.top_k)

        hits: List[ChunkOut] = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity_score = 1 - (distance / 2.0)
            hits.append(
                ChunkOut(
                    id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=round(similarity_score, 4)
                )
            )

        return QueryResponse(query=req.query, hits=hits, total_hits=len(hits))

    # -------------------------------
    # RAG Query (Retrieval + Generation)
    # -------------------------------

    def _get_llm(self) -> ChatGroq:
        """Create or return the Groq LLM. Requires GROQ_API_KEY when first used."""
        if self._llm is not None:
            return self._llm
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set GROQ_API_KEY in the environment or .env to use RAG query (LLM) features."
            )
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=True,
            api_key=settings.groq_api_key,
        )
        return self._llm

    def rag_query(self, query: str, top_k: int = 5, stream: bool = False) -> Dict[str, Any]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        callback_handler = StreamingCallbackHandler() if stream else None

        qa_chain = RetrievalQA.from_chain_type(
            llm=self._get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            callbacks=[callback_handler] if callback_handler else None
        )

        result = qa_chain.invoke({"query": query})

        sources = []
        for i, doc in enumerate(result["source_documents"]):
            sources.append({
                "text_snippet": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "N/A"),
                "relevance_score": round(1.0 - (i * 0.1), 2) if i < 5 else 0.5
            })

        response = {
            "answer": result["result"],
            "sources": sources,
            "metadata": {
                "query": query,
                "top_k": top_k,
                "total_sources": len(sources),
                "execution_time_ms": round((time.time()) * 1000, 2),
                "model": "llama-3.3-70b-versatile"
            }
        }

        if stream and callback_handler:
            response["full_stream"] = callback_handler.get_full_response()

        return response

    # -------------------------------
    # Service Stats
    # -------------------------------

    def get_stats(self) -> Dict[str, Any]:
        repo_stats = self.repo.get_stats()
        return {
            "repository": repo_stats,
            "embedding_model": {
                "name": settings.embedding_model,
                "dimension": self.embedder_model.get_sentence_embedding_dimension()
            },
            "llm": {
                "model": "llama-3.3-70b-versatile",
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens
            },
            "chunking": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
        }
