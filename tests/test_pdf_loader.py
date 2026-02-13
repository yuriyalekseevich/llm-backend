"""
Simple test for PyPDFLoader.
Run: python -m tests.week6.test_pdf_loader
"""

# ==== FIX PYTHON PATH - ADD THIS AT THE VERY TOP ====
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent  # Goes from tests/ -> backend/
sys.path.insert(0, str(backend_dir))
# ====================================================

from app.utils.logging_config import logger, setup_logging
import os
import shutil
import logging
import tempfile
from reportlab.lib.pagesizes import letter
from langchain_community.document_loaders import PyPDFLoader
from reportlab.pdfgen import canvas

setup_logging(level=logging.INFO)

def create_test_pdf() -> Path:
    """Create a simple test PDF file and return Path object."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    # Create a simple PDF
    c = canvas.Canvas(tmp_path, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF document.")
    c.drawString(100, 730, "It contains two lines of text.")
    c.save()
    
    return Path(tmp_path)  # ← RETURN PATH, NOT STRING!

def test_pdf_loader():
    """Test PyPDFLoader with a real PDF."""
    print("\n📄 Testing PyPDFLoader...")
    sys.stdout.flush()  # Force print to show
    
    # Ask user which file to use
    print("\n📋 Choose PDF source:")
    print("   1. Create new test PDF (temp file)")
    print("   2. Use Downloads/Business Requirements PDF")
    print("   3. Specify custom path")
    sys.stdout.flush()
    
    choice = input("\nEnter choice (1-3): ").strip()
    print(f"✅ You chose: {choice}")
    sys.stdout.flush()
    
    # Initialize variables
    pdf_path = None
    should_cleanup = False
        
    if choice == "1":
        # Create temp file - NOW RETURNS PATH!
        pdf_path = create_test_pdf()
        should_cleanup = True
        logger.info(f"✅ Created temp PDF: {pdf_path}")
        
    elif choice == "2":
        # Use Downloads folder
        downloads_path = Path.home() / "Downloads"
        
        # Try different possible filenames
        possible_names = [
            "TRIAL AGREEMENT.pdf",
            "Mobile App Business Requirements.pdf",
            "Business Requirements.pdf"
        ]
        
        original = None
        for name in possible_names:
            test_path = downloads_path / name
            if test_path.exists():
                original = test_path
                print(f"✅ Found file: {original}")
                break
        
        if not original:
            # List all PDFs in Downloads to help user
            print("\n📂 PDF files in Downloads:")
            for pdf in downloads_path.glob("*.pdf"):
                print(f"   - {pdf.name}")
            
            logger.error(f"❌ File not found. Please check the filename above.")
            return None
            
        # Create copy
        temp_dir = Path(tempfile.gettempdir())
        pdf_path = temp_dir / f"test_copy_{original.name}"
        shutil.copy2(original, pdf_path)
        should_cleanup = True
        logger.info(f"✅ Created working copy: {pdf_path}")
        
    elif choice == "3":
        # Custom path
        custom = input("Enter full path to PDF: ").strip()
        pdf_path = Path(custom).expanduser()  # Handle ~ if present
        if not pdf_path.exists():
            logger.error(f"❌ File not found: {pdf_path}")
            return None
        should_cleanup = False  # Don't delete user's file
        logger.info(f"📂 Using user file: {pdf_path}")
        
    else:
        logger.error("❌ Invalid choice")
        return None
    
    # DEBUG: Check type of pdf_path
    print(f"🔍 pdf_path type: {type(pdf_path)}")
    sys.stdout.flush()
    
    try:
        # Initialize loader
        print(f"\n📂 Loading: {pdf_path}")
        sys.stdout.flush()
        
        loader = PyPDFLoader(str(pdf_path))  # Convert to string for loader
        logger.info("✅ PyPDFLoader initialized")
        
        # Load document
        documents = loader.load()
        logger.info(f"✅ PDF loaded successfully")
        
        # Show results
        print(f"\n   📚 Loaded {len(documents)} page(s)")
        for i, doc in enumerate(documents[:3]):  # Show first 3 pages
            print(f"\n   📄 Page {i+1}:")
            print(f"      Content: {doc.page_content[:50]}...")
            print(f"      Metadata: {doc.metadata}")
        
        if len(documents) > 3:
            print(f"      ... and {len(documents)-3} more pages")
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ PyPDFLoader failed: {e}")
        raise
    finally:
        # Clean up only if we should
        if should_cleanup and pdf_path:
            # Check if it's a Path object or string
            if isinstance(pdf_path, Path):
                file_exists = pdf_path.exists()
            else:
                file_exists = os.path.exists(pdf_path)
            
            if file_exists:
                try:
                    os.unlink(str(pdf_path))  # Convert to string for os.unlink
                    logger.info(f"🧹 Cleaned up test file: {pdf_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to clean up: {e}")
            else:
                logger.info(f"📁 File already gone: {pdf_path}")
        else:
            logger.info(f"📁 Preserved file: {pdf_path}")

if __name__ == "__main__":
    # Print current directory for debugging
    print(f"📂 Current directory: {Path.cwd()}")
    print(f"📂 Script directory: {Path(__file__).parent}")
    sys.stdout.flush()
    
    test_pdf_loader()