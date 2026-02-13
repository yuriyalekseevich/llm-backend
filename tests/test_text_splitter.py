"""
Simple test for RecursiveCharacterTextSplitter.
Run: python -m tests.week6.test_text_splitter
"""

# ==== FIX PYTHON PATH - ADD THIS AT THE VERY TOP ====
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent  # Goes from tests/ -> backend/
sys.path.insert(0, str(backend_dir))
# ====================================================

"""
Splitting text into chunks is a critical step in RAG pipelines. 
This test verifies that the RecursiveCharacterTextSplitter correctly splits a document into manageable chunks 
while respecting the specified chunk size and overlap. It also ensures that the splitting 
logic handles various separators effectively, which is essential for maintaining the 
coherence of the text and improving the quality of embeddings and search results.

For developers it is important to understand how the text splitter works and to be able to test it with real documents,
as the quality of chunking can significantly impact the performance of the RAG system.

It is better to see the text before the splitting process to set the splitter configurations correctly.

"""

from app.utils.logging_config import logger, setup_logging
import logging

setup_logging(level=logging.INFO)

def test_text_splitter():
    """Test RecursiveCharacterTextSplitter with sample text."""
    print("\n✂️ Testing RecursiveCharacterTextSplitter...")
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_classic.schema import Document
    
    # For test you can Create sample document

    # long_text = """
    # This is the first paragraph about AI and machine learning. 
    # Artificial Intelligence is transforming how we build software.
    # Large Language Models can understand and generate text.
    
    # This is the second paragraph about RAG systems.
    # Retrieval Augmented Generation combines search with LLMs.
    # It helps ground responses in factual information.
    
    # This is the third paragraph about vector databases.
    # ChromaDB is an open-source vector database.
    # It stores embeddings for efficient similarity search.
    # """

    # document = Document(
    #     page_content=long_text,
    #     metadata={"source": "test", "page": 1}
    # )

    # Or load a real PDF page for testing

    downloads_path = Path.home() / "Downloads/TRIAL AGREEMENT.pdf"

    pdf_path = Path(downloads_path)
    
    if not pdf_path.exists():
        print(f"⚠️ PDF not found at {pdf_path}")
        print("Please provide a valid PDF path")
        return
    
    # Load PDF
    print(f"📂 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    document = loader.load()
    
    print(f"📚 Loaded {len(document)} pages")
    for i, doc in enumerate(document[12:15]):  # Show lats 3 pages
        print(f"   Page {i+1}: {len(doc.page_content)} chars")
        print(f"   Metadata: {doc.metadata}")
    
    # Test TextSplitter after doc is created/loaded
    
    logger.info(f"📝 Original document: {len(doc.page_content)} chars")
    
    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[
        "\n\n",  # Double newline (rare in your case)
        "\n• ",  # Bullet points with space
        "\n•",   # Bullet points without space (just in case)
        "\n",    # Single newline
        ".  ",    # Period + 2spaces (end of sentences)
        ".",     # Period without space (if no space after)
        " ",     # Last resort
    ],
    keep_separator=True,
    strip_whitespace=True
    )

    logger.info("✅ TextSplitter initialized")
    
    # Split document
    chunks = splitter.split_documents([doc])
    
    print(f"   📚 Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:9]):
        # Clean the content for display
        preview = (chunk.page_content[:280]
                  .replace('\n', ' ')
                  .replace('\r', '')
                  .strip())
        preview = ' '.join(preview.split())  # Collapse multiple spaces
        
        print(f"   ✂️ Chunk {i+1}: {preview}...")
        
        # Show page number if available
        if hasattr(chunk, 'metadata') and 'page' in chunk.metadata:
            print(f"      📄 Page: {chunk.metadata['page']}")
    
    print("   " + "-" * 60)
    
    return chunks

if __name__ == "__main__":
    test_text_splitter()