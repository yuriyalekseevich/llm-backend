import chromadb
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Comment: Repository class for abstracting vector DB operations.
# Uses persistent storage to keep data between server restarts.
class ChromaRepository:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        logger.info(f"Initializing ChromaRepository with persist_dir={persist_dir} and collection={collection_name}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info("Existing Chroma collection loaded successfully")
        except ValueError:
            logger.warning("Chroma collection not found; creating new one")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine distance for normalized embeddings
            )
            logger.info("New Chroma collection created")

    # Comment: Add method for inserting chunks.
    # Logs the number added for console visibility.
    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        logger.info(f"Adding {len(ids)} documents to Chroma collection")
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Successfully added {len(ids)} documents")

    # Comment: Query method for vector search.
    # Returns raw results; formatting happens in service.
    def query(self, query_embedding: List[float], n_results: int = 3) -> dict:
        logger.info(f"Executing Chroma query with n_results={n_results}")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        logger.info(f"Query returned {len(results.get('ids', [[]])[0])} hits")
        return results

    # Comment: Utility to get current document count (for health checks).
    def count(self) -> int:
        count = self.collection.count()
        logger.debug(f"Current document count: {count}")
        return count