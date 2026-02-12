#!/usr/bin/env python3
"""
Week 6: Full RAG Pipeline — Best Test Ever.

This script demonstrates a production-style RAG pipeline:
  1. Load a PDF (e.g. Flutter Senior Interview Q&A).
  2. Chunk the document with overlap.
  3. Clean chunks (remove noise: extra whitespace, control chars, too-short fragments).
  4. Embed chunks and store in a vector DB (Chroma).
  5. Take a user question → turn it into a vector → similarity search.
  6. Return best-matching chunks (and optionally full RAG answer via LLM).

Every step is logged and explained. Uses a dedicated collection so the main app is untouched.

Usage:
  # Interactive: script loads PDF, then prompts for a question
  python week6_rag_test.py "/Users/yuriy/Downloads/Flutter Senior Interview Qa Russian.pdf"

  # With question on the command line
  python week6_rag_test.py "/path/to/file.pdf" "What is Flutter?"

  # Optional: top_k and skip LLM
  python week6_rag_test.py "/path/to/file.pdf" "Your question?" --top-k 5 --no-llm
"""

import argparse
import logging
import os
import re
import sys
from types import SimpleNamespace

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.repositories.chroma_repo import ChromaRepository
from app.services.rag_service import RagService
from app.models.rag import QueryRequest
from app.utils.logging_config import logger

# Default PDF path for Week 6 (Flutter Senior Interview Q&A Russian)
DEFAULT_PDF_PATH = "/Users/yuriy/Downloads/Flutter Senior Interview Qa Russian.pdf"
# Dedicated collection so we don't overwrite main app data
WEEK6_COLLECTION = "week6_flutter_qa"
# Minimum chunk length after cleaning (chunks shorter than this are dropped as "noise")
MIN_CHUNK_LEN = 20

# Ensure we see step-by-step logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def clean_chunk_text(text: str) -> str:
    """
    Remove noise from chunk text so embeddings and search are cleaner.
    - Normalize whitespace (collapse multiple spaces/newlines to single space).
    - Remove control characters (keep tab, newline, carriage return).
    - Strip leading/trailing whitespace.
    """
    if not text or not text.strip():
        return ""
    # Replace any run of whitespace (including newlines) with a single space
    normalized = re.sub(r"\s+", " ", text)
    # Remove control characters except \t \n \r
    cleaned = "".join(c for c in normalized if c in "\t\n\r" or ord(c) >= 32)
    return cleaned.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 RAG test: load PDF, chunk & clean, embed, answer user question with vector search."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=DEFAULT_PDF_PATH,
        help=f"Path to PDF (default: {DEFAULT_PDF_PATH})",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to answer from the PDF (if omitted, prompt interactively)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only run vector search; do not call LLM for a generated answer",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Chunk size in chars (default: from config, {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"Chunk overlap (default: from config, {settings.chunk_overlap})",
    )
    args = parser.parse_args()

    pdf_path = os.path.expanduser(args.pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    # ----- Step 1: Load PDF -----
    print("\n" + "=" * 70)
    print("STEP 1: Load PDF")
    print("=" * 70)
    print(f"Path: {pdf_path}")
    logger.info("Loading PDF for Week 6 RAG test", extra={"path": pdf_path})

    embedder = SentenceTransformer(settings.embedding_model)
    repo = ChromaRepository(collection_name=WEEK6_COLLECTION)
    service = RagService(embedder, repo)

    docs = service.load_pdf(pdf_path)
    print(f"✅ Loaded {len(docs)} page(s)")

    # ----- Step 2: Chunk documents -----
    print("\n" + "=" * 70)
    print("STEP 2: Chunk documents")
    print("=" * 70)
    print("Splitting by paragraphs/lines/sentences so each chunk fits embedding + context.")
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap
    print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    raw_chunks = service.chunk_docs(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"✅ Raw chunks before cleaning: {len(raw_chunks)}")

    # ----- Step 3: Clean chunks (remove noise) -----
    print("\n" + "=" * 70)
    print("STEP 3: Clean chunks (remove noise)")
    print("=" * 70)
    print("Normalizing whitespace, stripping control chars, dropping too-short fragments.")

    cleaned_chunks = []
    dropped = 0
    for i, chunk in enumerate(raw_chunks):
        text = getattr(chunk, "page_content", str(chunk))
        cleaned = clean_chunk_text(text)
        if len(cleaned) < MIN_CHUNK_LEN:
            dropped += 1
            logger.debug("Dropped short chunk", extra={"index": i, "len": len(cleaned)})
            continue
        cleaned_chunks.append(
            SimpleNamespace(
                page_content=cleaned,
                metadata=getattr(chunk, "metadata", {}).copy(),
            )
        )

    logger.info(
        "Chunks after cleaning",
        extra={"kept": len(cleaned_chunks), "dropped": dropped},
    )
    print(f"✅ Kept {len(cleaned_chunks)} chunks (dropped {dropped} too short or empty)")

    if not cleaned_chunks:
        print("❌ No chunks left after cleaning. Check PDF content and MIN_CHUNK_LEN.")
        sys.exit(1)

    # ----- Step 4: Ingest into vector store -----
    print("\n" + "=" * 70)
    print("STEP 4: Ingest chunks into vector store")
    print("=" * 70)
    print("Embedding each chunk and storing in Chroma (cosine similarity).")

    source_name = os.path.basename(pdf_path)
    added = service.ingest_chunks(cleaned_chunks, source=source_name)
    print(f"✅ Ingested {added} chunks into collection '{WEEK6_COLLECTION}'")

    # ----- Step 5: Get user question -----
    print("\n" + "=" * 70)
    print("STEP 5: User question")
    print("=" * 70)

    question = args.question
    if not question:
        question = input("Enter your question (about the PDF): ").strip()
    if not question:
        print("No question provided. Exiting.")
        sys.exit(0)

    print(f"Question: «{question}»")

    # ----- Step 6: Turn question into vector and search -----
    print("\n" + "=" * 70)
    print("STEP 6: Vector search")
    print("=" * 70)
    print("Encoding the question into the same embedding space and finding nearest chunks.")

    req = QueryRequest(query=question, top_k=args.top_k)
    response = service.search(req)

    print(f"✅ Found {len(response.hits)} hit(s) (best first by similarity).")
    print()
    for i, hit in enumerate(response.hits, 1):
        print(f"--- Hit {i} (score={hit.score:.4f}, higher = more similar) ---")
        print(f"  ID: {hit.id}")
        print(f"  Source: {hit.metadata.get('source', 'N/A')} (page: {hit.metadata.get('page', 'N/A')})")
        snippet = hit.text[:400] + "..." if len(hit.text) > 400 else hit.text
        print(f"  Text: {snippet}")
        print()

    # ----- Step 7 (optional): Full RAG answer via LLM -----
    if not args.no_llm:
        print("=" * 70)
        print("STEP 7: RAG answer (retrieve + LLM)")
        print("=" * 70)
        try:
            result = service.rag_query(query=question, top_k=args.top_k)
            print("Answer:", result.get("answer", "(none)"))
            print("\nSources:")
            for s in result.get("sources", [])[:5]:
                print(f"  - {s.get('source', '')} p.{s.get('page', '')}: {s.get('text_snippet', '')[:150]}...")
        except ValueError as e:
            if "GROQ_API_KEY" in str(e):
                print("(Skipping LLM: GROQ_API_KEY not set. Use --no-llm to avoid this message.)")
            else:
                raise
    else:
        print("(Skipping LLM step; use without --no-llm for full RAG answer.)")

    print("\n" + "=" * 70)
    print("Week 6 RAG test finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
