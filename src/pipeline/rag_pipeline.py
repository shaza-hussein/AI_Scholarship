"""
Dual-Store RAG Ingestion Pipeline.

This module is responsible for taking the cleaned master JSON data and building
the two foundational databases for the AI Scholarship system:
1. DuckDB (Relational Store): For deterministic filtering and fast metadata queries.
2. ChromaDB (Vector Store): For semantic search and embedding storage.

This script should be executed whenever the underlying scholarship data is updated.
"""

import os
import sys
import logging
import pandas as pd
import duckdb
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Robust path resolution based on the absolute file location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.rag.document_chunker import HybridDocumentChunker

# Configure system logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class RAGPipelineBuilder:
    """
    Production pipeline orchestrator for the Dual-Store architecture.
    
    This class handles the end-to-end process of reading JSON data, optimizing it,
    chunking it according to heuristically determined hyperparameters, and saving it
    into both analytical and vector databases.
    """

    def __init__(self, json_path: str, vector_db_path: str, relational_db_path: str):
        """
        Initializes the pipeline builder with file paths and optimized hyperparameters.

        Args:
            json_path (str): Absolute path to the cleaned master JSON file.
            vector_db_path (str): Target directory for saving the ChromaDB instances.
            relational_db_path (str): Target file path for the DuckDB analytical database.
        """
        self.json_path = json_path
        self.vector_db_path = vector_db_path
        self.relational_db_path = relational_db_path
        
        # Hyperparameters determined via empirical testing to preserve semantic integrity
        self.chunk_size = 800
        self.chunk_overlap = 100
        self.min_chunk_length = 30
        
        # Multilingual embedding model capable of handling English and German natively
        self.embedding_model_name = "BAAI/bge-m3"

    def build_relational_store(self):
        """
        Constructs the DuckDB analytical database.
        
        Process:
        1. Loads the JSON file into a Pandas DataFrame.
        2. Drops redundant or purely operational columns to save memory and optimize query speed.
        3. Persists the optimized DataFrame into a DuckDB file.
        """
        logging.info("[1/2] Building Relational Store (DuckDB)...")
        try:
            df = pd.read_json(self.json_path)
            
            # Columns to drop to optimize analytical database storage
            columns_to_drop = [
                'country', 'degree_level', 'funding_type', 
                'deadline', 'tags', 'parsed_date', 'is_expired'
            ]
            
            # Safely drop columns only if they exist in the dataframe
            existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=existing_columns_to_drop)
            
            logging.info("Dropped %d redundant columns before storage.", len(existing_columns_to_drop))
            
            os.makedirs(os.path.dirname(self.relational_db_path), exist_ok=True)
            
            # Create a persistent DuckDB connection and ingest the dataframe
            conn = duckdb.connect(self.relational_db_path)
            conn.execute("CREATE OR REPLACE TABLE scholarships AS SELECT * FROM df")
            conn.close()
            
            logging.info("Success: Relational database saved to %s", self.relational_db_path)
        except Exception as e:
            logging.error("Error building relational store: %s", str(e))
            raise

    def build_vector_store(self):
        """
        Constructs the ChromaDB semantic vector database.
        
        Process:
        1. Chunks the raw documents using the HybridDocumentChunker.
        2. Initializes the BAAI/bge-m3 embedding model (utilizing GPU if available).
        3. Ingests the chunked documents into ChromaDB in batches to prevent memory overflow.
        4. Persists the vector store to disk.
        """
        logging.info("[2/2] Building Vector Store (ChromaDB)...")
        try:
            chunker = HybridDocumentChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_length=self.min_chunk_length
            )
            
            logging.info("Loading and chunking documents...")
            base_docs = chunker.load_and_filter(self.json_path)
            final_chunks = chunker.process_documents(base_docs)
            logging.info("Total valid chunks generated: %d", len(final_chunks))
            
            logging.info("Loading embedding model: %s...", self.embedding_model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logging.info("Initializing ChromaDB connection...")
            os.makedirs(self.vector_db_path, exist_ok=True)
            
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=self.vector_db_path
            )
            
            logging.info("Ingesting chunks into ChromaDB (Batch processing)...")
            batch_size = 500
            
            # Process in batches to manage RAM/VRAM constraints during vectorization
            for i in tqdm(range(0, len(final_chunks), batch_size), desc="Ingesting to ChromaDB"):
                batch = final_chunks[i:i + batch_size]
                vector_store.add_documents(documents=batch)
                
            logging.info("Success: Vector database saved to %s", self.vector_db_path)
            
        except Exception as e:
            logging.error("Error building vector store: %s", str(e))
            raise

    def run(self):
        """
        Executes the full dual-store pipeline synchronously.
        """
        logging.info("Starting Dual-Store Pipeline execution...")
        self.build_relational_store()
        self.build_vector_store()
        logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    
    JSON_DATA_PATH = os.path.join(BASE_DIR, "data_Json", "processed", "master_scholarships_clean.json")
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
    RELATIONAL_DB_FILE = os.path.join(BASE_DIR, "db", "analytical.duckdb")
    
    pipeline = RAGPipelineBuilder(
        json_path=JSON_DATA_PATH,
        vector_db_path=VECTOR_DB_DIR,
        relational_db_path=RELATIONAL_DB_FILE
    )
    
    pipeline.run()