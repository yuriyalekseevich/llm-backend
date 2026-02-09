import logging
import os
from typing import List
import uuid

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import; requires pip install langchain-text-splitters
from app.repositories.chroma_repo import ChromaRepository
from app.services.rag_service import RagService
from app.models.rag import ChunkIn, QueryRequest
from app.utils.logging_config import logger  # Assuming your logger setup

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🧪 Advanced Vector Embeddings & RAG Search Test")
print("   - This script builds on the basic test to demonstrate a full RAG-like workflow.")
print("   - We'll load a sample document (create a text file for this), chunk it into pieces,")
print("   - embed each chunk, store in Chroma, and perform similarity searches.")
print("   - Explanations are printed at each step for learning.")
print("   - Vectors are shown truncated (first 5 dims) for brevity—full vectors are 384D.")
print("=" * 80)

# Step 0: Setup - Create a sample document if it doesn't exist
# Explanation: For real RAG, we'd load a PDF or doc. Here, we create a simple text file as our "document".
# You can replace this with a real file, e.g., via PyPDFLoader as in Week 6.
SAMPLE_FILE = "sample_doc.txt"
if not os.path.exists(SAMPLE_FILE):
    with open(SAMPLE_FILE, "w") as f:
        f.write("""
Flutter Documentation Excerpt:

Flutter is Google's UI toolkit for building beautiful, natively compiled applications for mobile, web, desktop, and embedded devices from a single codebase. Flutter works with existing code, is used by developers and organizations around the world, and is free and open source.

Key Features:
- Fast development: Hot reload allows you to quickly and easily experiment, build UIs, add features, and fix bugs.
- Expressive and flexible UI: Quickly ship features with a focus on native end-user experiences.
- Native performance: Flutter’s widgets incorporate all critical platform differences such as scrolling, navigation, icons, and fonts.

AI Embeddings Basics:
Embeddings are dense vector representations of text, images, or other data that capture semantic meaning. In NLP, models like Sentence Transformers convert sentences into fixed-length vectors (e.g., 384 dimensions) where similar meanings are close in vector space.

Similarity Search:
We use cosine similarity to find how close two vectors are: cos(theta) = (A · B) / (|A| |B|). Scores near 1 mean very similar; near 0 mean unrelated.
        """)
    print(f"✅ Created sample document: {SAMPLE_FILE}")
else:
    print(f"✅ Sample document already exists: {SAMPLE_FILE}")

# Step 1: Load the Document
# Explanation: Read the file content. In production, use loaders like PyPDFLoader for PDFs.
try:
    print("\n📄 Step 1: Loading the document...")
    with open(SAMPLE_FILE, "r") as f:
        doc_text = f.read().strip()
    print(f"✅ Document loaded! Length: {len(doc_text)} chars")
    print("Preview:", doc_text[:200] + "...")  # Show snippet
except Exception as e:
    print(f"❌ Failed to load document: {e}")
    raise

# Step 2: Chunk the Document
# Explanation: Split text into smaller chunks (e.g., 200-500 chars) for better embedding granularity.
# Overlap helps retain context across chunks. This prevents exceeding context limits in LLMs.
try:
    print("\n✂️ Step 2: Chunking the document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Target chunk size
        chunk_overlap=50,  # Overlap for context
        separators=["\n\n", "\n", " ", ""]  # Split on paragraphs, lines, etc.
    )
    chunks = text_splitter.split_text(doc_text)
    print(f"✅ Chunked into {len(chunks)} pieces!")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (len={len(chunk)}): {chunk[:100]}...")
except Exception as e:
    print(f"❌ Failed to chunk: {e}")
    raise

# Step 3: Load Embedder
# Explanation: SentenceTransformer creates vector embeddings. 'all-MiniLM-L6-v2' is efficient (384 dims).
try:
    print("\n📦 Step 3: Loading SentenceTransformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")  # 'mps' for Mac M1+, 'cuda' for GPU
    print(f"✅ Embedder loaded! Dimension: {embedder.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"❌ Failed to load embedder: {e}")
    raise

# Step 4: Generate Embeddings for Chunks
# Explanation: Convert each chunk to a vector. Similar texts get similar vectors.
# We normalize for cosine similarity (scores 0-1, lower distance = more similar).
try:
    print("\n🔍 Step 4: Generating embeddings for chunks...")
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    print(f"✅ Embeddings generated! Shape: {embeddings.shape} (chunks x dims)")
    for i, emb in enumerate(embeddings):
        # Show truncated vector (first 5 values)
        print(f"Chunk {i+1} vector preview: {emb[:5].tolist()}...")
except Exception as e:
    print(f"❌ Failed to generate embeddings: {e}")
    raise

# Step 5: Prepare Chunks for Storage
# Explanation: Create ChunkIn objects with unique IDs and metadata (e.g., source file).
try:
    print("\n📦 Step 5: Preparing ChunkIn objects...")
    chunk_ins: List[ChunkIn] = []
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        chunk_ins.append(
            ChunkIn(
                id=str(uuid.uuid4()),  # Unique ID
                text=text,
                metadata={"source": SAMPLE_FILE, "chunk_index": i}
            )
        )
    print(f"✅ Prepared {len(chunk_ins)} chunks for storage!")
except Exception as e:
    print(f"❌ Failed to prepare chunks: {e}")
    raise

# Step 6: Initialize Chroma Repository
# Explanation: Chroma is our vector DB. It stores embeddings with texts/IDs/metadata.
# Persistence in ./chroma_db means data survives restarts.
try:
    print("\n🗄️ Step 6: Initializing ChromaRepository...")
    chroma_repo = ChromaRepository()  # Uses defaults: persist_dir="./chroma_db", collection="rag_docs"
    print("✅ ChromaRepository initialized!")
except Exception as e:
    print(f"❌ Failed to init Chroma: {e}")
    raise

# Step 7: Add Chunks to Vector Store via RagService
# Explanation: RagService embeds (already done) and adds to repo. In full app, this is API-called.
try:
    print("\n➕ Step 7: Adding chunks to vector store...")
    rag_service = RagService(embedder, chroma_repo)
    count = rag_service.add_chunks(chunk_ins)
    print(f"✅ Added {count} chunks!")
    logger.info(f"Chunks added: {[c.id for c in chunk_ins]}")
except Exception as e:
    print(f"❌ Failed to add chunks: {e}")
    raise

# Step 8: Perform Similarity Searches
# Explanation: Embed a query, search DB for top_k similar chunks (by cosine distance).
# Lower score = more similar. Use these hits in RAG to augment LLM prompts.
try:
    print("\n🔎 Step 8: Testing searches...")
    queries = [
        "What is Flutter?",
        "Explain embeddings in AI.",
        "How does similarity search work?"
    ]
    for query_text in queries:
        print(f"\nQuery: '{query_text}'")
        query_req = QueryRequest(query=query_text, top_k=2)
        response = rag_service.search(query_req)
        print("✅ Search completed!")
        if not response.hits:
            print("⚠️ No results - try a semantically similar query.")
            continue
        for hit in response.hits:
            print(f"- ID: {hit.id}, Score: {hit.score:.4f} (lower = more similar)")
            print(f"  Text: {hit.text[:150]}...")  # Truncate
            print(f"  Metadata: {hit.metadata}")
            print("---")
except Exception as e:
    print(f"❌ Failed to search: {e}")
    raise

print("\n🚀 All advanced tests passed! Study the steps above.")
print(f"   - Document file: {os.path.abspath(SAMPLE_FILE)} (edit this for real content).")
print("   - Chroma DB stored in: ./chroma_db (persistent across runs).")
print("   - Next: Integrate with LLM for full RAG (Week 6).")