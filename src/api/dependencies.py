import logging
from src.rag.retriever import ScholarshipRetriever
from src.rag.generator import ScholarshipGenerator

logger = logging.getLogger(__name__)

# Global variables to hold the loaded instances
_retriever_instance = None
_generator_instance = None

def init_ai_engines():
    """Initializes the heavy AI models in memory once at startup."""
    global _retriever_instance, _generator_instance
    
    if _retriever_instance is None:
        logger.info("Loading Hybrid Retriever models into memory...")
        _retriever_instance = ScholarshipRetriever()
        
    if _generator_instance is None:
        logger.info("Loading AI Generator (Groq) into memory...")
        _generator_instance = ScholarshipGenerator()
        
    logger.info(" All AI Engines successfully loaded and ready.")

def get_retriever() -> ScholarshipRetriever:
    """Dependency injection function to get the retriever instance."""
    if _retriever_instance is None:
        raise RuntimeError("Retriever engine is not initialized.")
    return _retriever_instance

def get_generator() -> ScholarshipGenerator:
    """Dependency injection function to get the generator instance."""
    if _generator_instance is None:
        raise RuntimeError("Generator engine is not initialized.")
    return _generator_instance