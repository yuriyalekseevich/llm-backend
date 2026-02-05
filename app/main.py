from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import logging
from app.routers import api 

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("🚀 Starting RAG Backend for Week 5: Embeddings & Vector Search")

# ChromaDB
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    logger.info("✅ ChromaDB initialized")
except Exception as e:
    logger.error(f"❌ ChromaDB failed: {e}")
    raise

# Embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info(f"✅ Embedding model loaded (dimension: {embedding_model.get_sentence_embedding_dimension()})")
except Exception as e:
    logger.error(f"❌ Embedding model failed: {e}")
    raise

app = FastAPI(
    title="RAG Backend - Week 5",
    description="Embeddings & Vector Search for LLM Practicum",
    version="1.0.0"
)

class Document(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def root():
    return {
        "project": "LLM Practicum - Week 5",
        "goal": "Embeddings & Vector Search (RAG Basics)",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "add_documents": "POST /documents",
            "search": "POST /search"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chromadb": "ready",
        "embeddings": "ready",
        "documents_count": collection.count()
    }

@app.get("/info")
def info():
    return {
        "week": "Week 5: Embeddings & Vector Search",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": embedding_model.get_sentence_embedding_dimension(),
        "vector_db": "ChromaDB",
        "distance_metric": "cosine"
    }

@app.post("/documents")
def add_documents(documents: List[Document]):
    """Add text documents to vector database"""
    try:
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.text for doc in documents]
        embeddings = [embedding_model.encode(doc.text).tolist() for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents")
        
        return {
            "status": "success",
            "message": f"Added {len(documents)} documents",
            "count": len(documents),
            "ids": ids
        }
        
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(request: SearchRequest):
    """Search for similar documents using vector similarity"""
    try:
        query_embedding = embedding_model.encode(request.query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": float(results['distances'][0][i]),
                    "rank": i + 1
                })
        
        return {
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Mount routers (scalable: Add more like app.include_router(admin_router, prefix="/admin"))
app.include_router(api.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
