"""
ChromaDB repository with comprehensive logging and error handling.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import uuid
from app.config import settings
from app.utils.logging_config import logger
import traceback

class ChromaRepository:
    """Repository for ChromaDB operations with full observability."""
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            persist_directory: Override default persist directory
            collection_name: Override default collection name
        """
        self.persist_directory = persist_directory or settings.chroma_db_path
        self.collection_name = collection_name or settings.collection_name
        
        logger.info(
            "🚀 Initializing ChromaRepository",
            extra={
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name
            }
        )
        
        try:
            # Initialize client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.debug("ChromaDB client created")
            
            # Get or create collection
            self.collection = self._get_or_create_collection()
            
            logger.info(
                "✅ ChromaRepository initialized",
                extra={
                    "collection_name": self.collection_name,
                    "document_count": self.collection.count()
                }
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to initialize ChromaRepository",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"📚 Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection if doesn't exist
            logger.info(f"🆕 Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> int:
        """
        Add documents to ChromaDB with validation and logging.
        
        Returns:
            Number of documents added
        """
        logger.info(
            "📝 Adding documents to ChromaDB",
            extra={
                "count": len(documents),
                "embedding_dim": len(embeddings[0]) if embeddings else 0
            }
        )
        
        # Validate inputs
        if not all(len(lst) == len(documents) for lst in [ids, embeddings, metadatas]):
            raise ValueError("All input lists must have same length")
        
        # Generate IDs if not provided or empty
        final_ids = []
        for i, doc_id in enumerate(ids):
            if not doc_id:
                final_ids.append(str(uuid.uuid4()))
                logger.debug(f"Generated UUID for document {i}")
            else:
                final_ids.append(doc_id)
        
        try:
            # Add to collection
            self.collection.add(
                ids=final_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            new_count = self.collection.count()
            logger.info(
                "✅ Documents added successfully",
                extra={
                    "added": len(documents),
                    "total_documents": new_count
                }
            )
            
            return len(documents)
            
        except Exception as e:
            logger.error(
                "❌ Failed to add documents",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query ChromaDB with embeddings.
        
        Args:
            query_embeddings: List of embedding vectors
            n_results: Number of results to return
            where: Optional filter conditions
            
        Returns:
            Query results with ids, distances, documents, metadatas
        """
        logger.info(
            "🔍 Querying ChromaDB",
            extra={
                "n_results": n_results,
                "has_filter": where is not None
            }
        )
        
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            
            # Log result statistics
            result_count = len(results['ids'][0]) if results['ids'] else 0
            logger.info(
                "✅ Query completed",
                extra={
                    "results_found": result_count,
                    "query_id": str(uuid.uuid4())[:8]  # For traceability
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "❌ Query failed",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def delete_collection(self) -> bool:
        """Delete the current collection (useful for testing)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.warning(f"🗑️ Deleted collection: {self.collection_name}")
            
            # Recreate empty collection
            self.collection = self.client.create_collection(self.collection_name)
            logger.info("🆕 Recreated empty collection")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            # Get sample to determine embedding dimension
            sample = None
            embedding_dim = 384  # Default for all-MiniLM-L6-v2
            if count > 0:
                sample = self.collection.get(limit=1)
                if sample and sample['embeddings']:
                    embedding_dim = len(sample['embeddings'][0])
            
            stats = {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": embedding_dim,
                "persist_directory": self.persist_directory
            }
            
            logger.info("📊 Collection stats", extra=stats)
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}