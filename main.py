from fastapi import FastAPI, Request
import time
import uuid
import logging
from logging_config import setup_logging, get_logger
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers import api  

load_dotenv()  # Load .env (OPENAI_API_KEY)

# Configure structured JSON logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="LLM Practicum Backend", version="0.1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info("request.start", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query": str(request.url.query),
    })
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request.error", extra={"request_id": request_id})
        raise
    duration = int((time.time() - start) * 1000)
    logger.info("request.end", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": duration,
    })
    response.headers["X-Request-ID"] = request_id
    return response

# CORS for Flutter (allow all origins for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:your_flutter_port"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers (scalable: Add more like app.include_router(admin_router, prefix="/admin"))
app.include_router(api.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # Note: disable `reload` when running programmatically to avoid uvicorn warning
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)