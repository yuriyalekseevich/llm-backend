from fastapi import FastAPI
from dotenv import load_dotenv
import logging
from app.routers.api import router as api_router
from app.routers.rag import router as rag_router
from app.routers.general import router as general_router  # New general router
# If you have other routers, import them here (e.g., from app.routers import api)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("🚀 Starting RAG Backend for Week 5: Embeddings & Vector Search")

# Create FastAPI app
app = FastAPI(
    title="RAG Backend - Week 5",
    description="Embeddings & Vector Search for LLM Practicum",
    version="1.0.0"
)

# Include routers
app.include_router(general_router)  # Mounts general endpoints at root
app.include_router(rag_router)  # Mounts at /rag
app.include_router(api_router, prefix="/api")  # Mounts API endpoints at /api
# If you have api.router: app.include_router(api.router, prefix="/api")