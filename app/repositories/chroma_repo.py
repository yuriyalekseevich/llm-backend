# app/repositories/chroma_repo.py

import chromadb
from chromadb.api.models import Collection
from typing import List, Dict, Any
import uuid
import logging

from app.utils.logging_config import logger

class ChromaRepository:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_docs"
    ):
        logger.info(f"Initializing ChromaRepository with persist_dir={persist_dir} and collection={collection_name}")
        
        # Create persistent client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection (with embedding_function=None since we handle embeddings externally)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for normalized scores
        )
        logger.info(f"✅ Collection '{collection_name}' ready (created if not exists)")

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> int:
        """Add embedded chunks to the collection."""
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            count = len(ids)
            logger.info(f"Added {count} documents to collection")
            return count
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """Query the collection with embeddings."""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=include
            )
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying: {e}")
            raise

    def delete_collection(self):
        """Delete the entire collection (for testing/reset)."""
        self.client.delete_collection(self.collection.name)
        logger.warning(f"Deleted collection {self.collection.name}")