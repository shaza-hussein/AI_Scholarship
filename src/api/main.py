from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.dependencies import init_ai_engines
from src.api.routes import search, cv, sop, chat

from src.database.database import engine, Base
from src.database import models


# Lifespan event to manage resources on startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*60)
    print(" Starting ScholarAI API Server...")
    print(" Initializing heavy AI models (this takes ~15 seconds)...")
    print("="*60)
    
    # Load models into RAM
    init_ai_engines()
    
    yield # The server is now running and accepting requests
    
    # Cleanup on shutdown (if needed)
    print("\n Shutting down ScholarAI API Server...")

models.Base.metadata.create_all(bind=engine)

# Initialize the FastAPI App
app = FastAPI(
    title="ScholarAI API",
    description="Backend API for the AI Scholarship System",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(search.router)
app.include_router(cv.router)
app.include_router(sop.router)
app.include_router(chat.router)

# 1. Health Check Endpoint
@app.get("/health", tags=["System"])
def health_check():
    """Simple endpoint to verify the server and AI engines are running."""
    return {
        "status": "healthy",
        "message": "ScholarAI Engine is up and running!"
    }