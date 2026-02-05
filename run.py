import uvicorn
from app.utils.logging_config import setup_logging

if __name__ == "__main__":
    # Setup JSON logging
    setup_logging()
    
    print("🚀 Starting backend on http://0.0.0.0:8000")
    
    uvicorn.run(
        "app.main:app",  # "package.module:app"
        host="0.0.0.0",
        port=8000,
        reload=True,  # only for development
        log_level="info"
    )
