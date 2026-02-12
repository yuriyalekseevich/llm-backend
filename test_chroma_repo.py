# test_chroma_repo.py
from app.repositories.chroma_repo import ChromaRepository
from app.utils.logging_config import setup_logging, logger
import numpy as np

def test_chroma_repo():
    """Test ChromaRepository functionality."""
    setup_logging()
    logger.info("🧪 Testing ChromaRepository...")
    
    # Initialize repo
    repo = ChromaRepository()
    
    # Get initial stats
    stats = repo.get_stats()
    logger.info(f"Initial stats: {stats}")
    
    # Create test data
    test_docs = ["This is test document 1", "This is test document 2"]
    test_embeddings = np.random.rand(2, 384).tolist()  # Random 384-dim vectors
    test_ids = ["test1", "test2"]
    test_metadata = [{"source": "test"}, {"source": "test"}]
    
    # Add documents
    added = repo.add_documents(
        ids=test_ids,
        embeddings=test_embeddings,
        metadatas=test_metadata,
        documents=test_docs
    )
    assert added == 2, "Should add 2 documents"
    
    # Query
    query_embedding = np.random.rand(1, 384).tolist()
    results = repo.query(query_embedding, n_results=1)
    assert len(results['ids'][0]) == 1, "Should return 1 result"
    
    # Clean up
    repo.delete_collection()
    logger.info("✅ ChromaRepository tests passed")
    
    return True

if __name__ == "__main__":
    test_chroma_repo()