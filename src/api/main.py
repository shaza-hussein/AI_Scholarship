from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.dependencies import init_ai_engines
from src.api.routes import search, cv, sop, chat, profile

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


origins = [
    "http://localhost:3000",      # React default port
    "http://localhost:5173",      # Vite default port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"                           # Allow all during development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(search.router)
app.include_router(cv.router)
app.include_router(sop.router)
app.include_router(chat.router)
app.include_router(profile.router)

# 1. Health Check Endpoint
@app.get("/health", tags=["System"])
def health_check():
    """Simple endpoint to verify the server and AI engines are running."""
    return {
        "status": "healthy",
        "message": "ScholarAI Engine is up and running!"
    }