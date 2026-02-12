# test_config.py
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration loading."""
    logger.info("🔧 Testing configuration...")
    
    assert settings.chunk_size == 300, "Chunk size mismatch"
    assert settings.collection_name == "rag_docs", "Collection name mismatch"
    assert settings.chroma_db_path == "./chroma_db", "DB path mismatch"
    
    logger.info(f"✅ Configuration valid:")
    logger.info(f"  - Model: {settings.embedding_model}")
    logger.info(f"  - Chunk size: {settings.chunk_size}")
    logger.info(f"  - DB path: {settings.chroma_db_path}")
    
    return True

if __name__ == "__main__":
    test_config()