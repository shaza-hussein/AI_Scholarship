"""
Hybrid Retrieval Engine with Cross-Encoder Re-ranking.

This module implements an advanced Hierarchical Retrieval architecture:
1. Deterministic Filtering (via DuckDB)
2. Semantic Query Formulation
3. Constrained Vector Search (via ChromaDB - Bi-Encoder)
4. Precision Re-ranking (via Cross-Encoder)
"""

import os
import sys
import logging
import duckdb
import torch
from typing import List, Any
from sentence_transformers import CrossEncoder

# Ensure correct path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.schemas.student import StudentProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ScholarshipRetriever:
    """
    Advanced matching engine bridging deterministic profiles with semantic databases,
    enhanced by a Cross-Encoder scoring layer for maximum accuracy.
    """

    def __init__(self):
        logging.info("Initializing Hybrid Retriever and Models...")
        self.duckdb_path = os.path.join(PROJECT_ROOT, "db", "analytical.duckdb")
        self.chroma_path = os.path.join(PROJECT_ROOT, "db", "chroma_db")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Bi-Encoder for initial dense retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        
        # 2. Load Cross-Encoder for precision re-ranking
        # Using the base version of BGE reranker for a balance of speed and accuracy
        self.reranker_model_name = "BAAI/bge-reranker-base"
        logging.info("Loading Cross-Encoder Re-ranker: %s", self.reranker_model_name)
        self.reranker = CrossEncoder(self.reranker_model_name, max_length=512, device=device)
        
        logging.info("Hybrid Retriever ready.")

    # def _deterministic_filter(self, profile: StudentProfile) -> List[str]:
    #     conn = duckdb.connect(self.duckdb_path)
    #     try:
    #         # We filter by nationality and degree level. We also include 'International' options.
    #         query = """
    #             SELECT scholarship_name 
    #             FROM scholarships 
    #             WHERE (eligible_nationality ILIKE ? OR eligible_nationality ILIKE '%International%')
    #             AND academic_level ILIKE ?
    #         """
    #         params = [f"%{profile.nationality}%", f"%{profile.academic_level}%"]
    #         results = conn.execute(query, params).fetchall()
    #         valid_names = [row[0] for row in results]
            
    #         logging.info("DuckDB Filtering: %d scholarships passed the hard criteria.", len(valid_names))
    #         return valid_names
    #     finally:
    #         conn.close()
    def _deterministic_filter(self, profile: StudentProfile) -> List[str]:
        """
        Stage 1: Hard Elimination via DuckDB with Exact Ontological Mapping.
        Utilizes precise IN operators based on the exact data distribution 
        for maximum query performance.
        """
        conn = duckdb.connect(self.duckdb_path)
        try:
            # Exact mapping based on actual database distribution
            level_map = {
                "Bachelor": ["Undergraduates", "Unspecified"],
                "Master": ["Graduates", "Unspecified"],
                "PhD": ["Doctoral/PhD", "Unspecified"],
                "Postdoc": ["Doctoral/PhD", "Unspecified"] 
            }
            
            # Fetch the precise database terms
            db_levels = level_map.get(profile.academic_level, ["Unspecified"])
            
            # Dynamically build placeholders for the IN operator (e.g., (?, ?))
            level_placeholders = ", ".join(["?"] * len(db_levels))
            
            # Build the query combining ILIKE for messy nationalities and IN for strict levels
            query = f"""
                SELECT scholarship_name 
                FROM scholarships 
                WHERE (
                    eligible_nationality ILIKE ? 
                    OR eligible_nationality ILIKE '%International%' 
                    OR eligible_nationality ILIKE '%Unspecified%'
                    OR eligible_nationality ILIKE '%All%'
                    OR eligible_nationality ILIKE '%World%'
                )
                AND academic_level IN ({level_placeholders})
            """
            
            # Safely parameterize
            params = [f"%{profile.nationality}%"] + db_levels
            
            results = conn.execute(query, params).fetchall()
            valid_names = [row[0] for row in results]
            
            logging.info("DuckDB Filtering: %d scholarships passed the hard criteria.", len(valid_names))
            return valid_names
        finally:
            conn.close()


    def _formulate_query(self, profile: StudentProfile) -> str:
        parts = [f"Student majoring in {profile.academic_major}."]
        if profile.research_interests:
            parts.append(f"Key research interests and focus: {profile.research_interests}.")
        if profile.skills:
            skills_str = ", ".join(profile.skills)
            parts.append(f"Possesses technical and academic skills in: {skills_str}.")
            
        formulated_query = " ".join(parts)
        logging.info("Formulated Query: '%s'", formulated_query)
        return formulated_query

    def match_scholarships(self, profile: StudentProfile, top_k: int = 5, fetch_k: int = 15) -> List[Any]:
        logging.info("Starting matching process for student from %s aiming for %s.", 
                     profile.nationality, profile.academic_level)
                     
        # Stage 1: Deterministic Filtering
        valid_scholarships = self._deterministic_filter(profile)
        if not valid_scholarships:
            logging.warning("Matching aborted: No scholarships found matching strict criteria.")
            return []
            
        search_query = self._formulate_query(profile)
        search_filter = {"scholarship_name": {"$in": valid_scholarships}}
        
        # Stage 2: Initial Dense Retrieval (Fetch more candidates than needed)
        try:
            logging.info("Executing ChromaDB Vector Search (Fetching Top-%d)...", fetch_k)
            initial_results = self.vector_store.similarity_search(
                query=search_query,
                k=fetch_k,
                filter=search_filter
            )
            
            if not initial_results:
                return []

            # Stage 3: Cross-Encoder Re-ranking
            logging.info("Executing Cross-Encoder Re-ranking on %d candidates...", len(initial_results))
            
            # Formulate pairs for the Cross-Encoder: [Query, Document]
            pairs = [[search_query, doc.page_content] for doc in initial_results]
            
            # Predict similarity scores
            scores = self.reranker.predict(pairs)
            
            # Attach scores to the documents
            for doc, score in zip(initial_results, scores):
                doc.metadata['rerank_score'] = float(score)
                
            # Sort documents by the new precision score in descending order
            initial_results.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
            
            # Return only the highly precise Top-K
            final_results = initial_results[:top_k]
            logging.info("Successfully isolated Top-%d perfectly matched scholarships.", len(final_results))
            
            return final_results
            
        except Exception as e:
            logging.error("Error during search or re-ranking pipeline: %s", str(e))
            return []