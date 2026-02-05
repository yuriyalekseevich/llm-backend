# repositories/chroma_repo.py
import chromadb
from typing import List, Dict, Any


class ChromaRepository:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # cosine хорошо для нормализованных эмбеддингов
            )

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embedding: List[float], n_results: int = 3) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )