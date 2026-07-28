"""
Database Verification Script.

This script performs sanity checks on both the DuckDB analytical store and the 
ChromaDB vector store to ensure data integrity and retrieval capabilities before 
moving to the API and production phases.
"""

import os
import duckdb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Calculate project root dynamically (one level up from the 'tests' directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def verify_databases():
    """
    Executes a sequence of diagnostic tests on the dual-store databases.
    
    Tests include:
    - DuckDB: Connection success, record count validation, and data retrieval.
    - ChromaDB: Model loading, vector space connection, and semantic similarity search.
    """
    
    print(">>> 1. Verifying DuckDB (Relational Store) <<<")
    duckdb_path = os.path.join(PROJECT_ROOT, "db", "analytical.duckdb")
    
    try:
        # Establish connection and perform basic SQL operations to ensure integrity
        conn = duckdb.connect(duckdb_path)
        count = conn.execute("SELECT COUNT(*) FROM scholarships").fetchone()[0]
        sample_title = conn.execute("SELECT scholarship_name FROM scholarships LIMIT 1").fetchone()[0]
        
        print(f"DuckDB Status: SUCCESS")
        print(f"Total Records: {count}")
        print(f"Sample Record: {sample_title}\n")
        
        conn.close()
    except Exception as e:
        print(f"DuckDB Error: {e}\n")

    print(">>> 2. Verifying ChromaDB (Vector Store) <<<")
    chroma_path = os.path.join(PROJECT_ROOT, "db", "chroma_db")
    
    try:
        # Initialize the same embedding model used during ingestion (CPU only for testing)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Connect to the persistent ChromaDB directory
        vector_store = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        
        # Perform a semantic search to verify that the vector space correctly interprets intent
        query = "Master degree in Computer Science in Germany"
        results = vector_store.similarity_search(query, k=2)
        
        print(f"ChromaDB Status: SUCCESS")
        print(f"Test Query: '{query}'")
        print(f"Top Result Metadata: {results[0].metadata['scholarship_name']}")
        print(f"Top Result Content Snippet: {results[0].page_content[:150]}...")
        
    except Exception as e:
        print(f"ChromaDB Error: {e}")


if __name__ == "__main__":
    verify_databases()