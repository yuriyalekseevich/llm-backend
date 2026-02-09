import logging
from sentence_transformers import SentenceTransformer
from app.repositories.chroma_repo import ChromaRepository
from app.services.rag_service import RagService
from app.models.rag import ChunkIn, QueryRequest
from app.utils.logging_config import logger  # Assuming this is your logger setup

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 50)
print("🧪 Vector Embeddings & Search Test")
print("=" * 50)

# Test 1: Load Embedder
try:
    print("\n📦 Loading SentenceTransformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")  # Use 'cuda' if GPU available
    print(f"✅ Embedder loaded! Dimension: {embedder.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ Failed to load embedder: {e}")
    raise

# Test 2: Generate Embeddings
try:
    print("\n🔍 Generating sample embeddings...")
    texts = [
        "Flutter is a UI toolkit for building natively compiled applications.",
        "Embeddings capture semantic meaning in vector form.",
        "Large Language Models are trained on vast text data."
    ]
    embeddings = embedder.encode(texts)
    print(f"✅ Embeddings generated! Shape: {embeddings.shape}")
    # Log first embedding snippet
    logger.info(f"First embedding preview: {embeddings[0][:5]}...")
except Exception as e:
    print(f"❌ Failed to generate embeddings: {e}")
    raise

# Test 3: Initialize Chroma Repository
try:
    print("\n🗄️ Initializing ChromaRepository...")
    chroma_repo = ChromaRepository()  # Assumes your init logic
    print("✅ ChromaRepository initialized!")
except Exception as e:
    print(f"❌ Failed to init Chroma: {e}")
    raise

# Test 4: Add Chunks to Vector Store
try:
    print("\n➕ Adding sample chunks...")
    chunks = [
        ChunkIn(id="test_chunk1", text=texts[0], metadata={"source": "flutter_docs"}),
        ChunkIn(id="test_chunk2", text=texts[1], metadata={"source": "ai_guide"}),
        ChunkIn(id="test_chunk3", text=texts[2], metadata={"source": "ai_guide"})
    ]
    rag_service = RagService(embedder, chroma_repo)
    count = rag_service.add_chunks(chunks)
    print(f"✅ Added {count} chunks!")
    logger.info(f"Chunks added: {[c.id for c in chunks]}")
except Exception as e:
    print(f"❌ Failed to add chunks: {e}")
    raise

# Test 5: Perform Search
try:
    print("\n🔎 Testing search...")
    query_req = QueryRequest(query="What are embeddings?", top_k=2)
    response = rag_service.search(query_req)
    print("✅ Search completed!")
    print("\nSearch Results:")
    for hit in response.hits:
        print(f"- ID: {hit.id}, Score: {hit.score:.4f}")
        print(f"  Text: {hit.text[:100]}...")  # Truncate for console
        print(f"  Metadata: {hit.metadata}")
        print("---")
    if not response.hits:
        print("⚠️ No results found - check if chunks match query semantically.")
except Exception as e:
    print(f"❌ Failed to search: {e}")
    raise

print("\n🚀 All tests passed if you see green checks! Check logs for details.")