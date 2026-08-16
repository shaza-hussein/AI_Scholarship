"""
Advanced Hybrid Retrieval Engine (Dense + Sparse + Cross-Encoder).

Architecture Pipeline:
1. Deterministic Filtering (DuckDB)
2. Hybrid Retrieval:
   - Dense Search (ChromaDB vector similarity)
   - Sparse Search (BM25 lexical keyword matching)
3. RRF Fusion (Reciprocal Rank Fusion)
4. Cross-Encoder Re-ranking with Confidence Threshold
"""

import os
import sys
import logging
import duckdb
import torch
from typing import List, Any, Dict

# Ensure correct path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.schemas.student import StudentProfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ScholarshipRetriever:
    def __init__(self):
        logging.info("Initializing Advanced Hybrid Retriever...")
        self.duckdb_path = os.path.join(PROJECT_ROOT, "db", "analytical.duckdb")
        self.chroma_path = os.path.join(PROJECT_ROOT, "db", "chroma_db")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Bi-Encoder (Dense)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        
        # 2. Build BM25 Index (Sparse) from ChromaDB data
        logging.info("Building BM25 Lexical Index in memory...")
        try:
            db_data = self.vector_store.get(include=['documents', 'metadatas'])
            docs = [Document(page_content=doc, metadata=meta) 
                    for doc, meta in zip(db_data['documents'], db_data['metadatas'])]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            logging.info("BM25 Index built successfully with %d chunks.", len(docs))
        except Exception as e:
            logging.error("Failed to build BM25: %s", str(e))
            self.bm25_retriever = None

        # 3. Load Cross-Encoder
        self.reranker_model_name = "BAAI/bge-reranker-base"
        self.reranker = CrossEncoder(self.reranker_model_name, max_length=512, device=device)
        logging.info("Advanced Hybrid Retriever ready.")

    def _detect_intents(self, query: str) -> dict:
        """
        Analyzes the user query to extract specific constraints and behavioral intents.
        Identifies requests for specific funding types and detects if the user is asking
        about a specific well-known scholarship by name to bypass restrictive filters.

        Args:
            query (str): The raw input query from the user.

        Returns:
            dict: A dictionary containing boolean flags and extracted constraints
                  (e.g., bypass_filters, target_funding).
        """
        query_lower = query.lower()
        intents = {
            "bypass_filters": False,
            "target_funding": None
        }

        specific_entities = ["daad", "erasmus", "chevening", "fulbright", "eiffel", "mext"]
        if any(name in query_lower for name in specific_entities):
            intents["bypass_filters"] = True

        if any(phrase in query_lower for phrase in ["fully funded", "full funding", "fully-funded", "100% funded", "ممولة بالكامل"]):
            intents["target_funding"] = "Fully Funded"
        elif any(phrase in query_lower for phrase in ["partially funded", "partial funding", "ممولة جزئيا"]):
            intents["target_funding"] = "Partially Funded"

        return intents

    def _deterministic_filter(self, profile: StudentProfile, intents: dict) -> List[str]:
        """
        Executes a deterministic SQL query against DuckDB to retrieve valid scholarship names.
        Always filters out 'Expired' status. Dynamically applies nationality, academic level,
        and funding category filters based on the detected intents.

        Args:
            profile (StudentProfile): The active user profile containing academic and demographic data.
            intents (dict): The parsed intents dictating which constraints to apply.

        Returns:
            List[str]: A list of valid scholarship names that pass the deterministic constraints.
        """
        conn = duckdb.connect(self.duckdb_path)
        try:
            base_query = "SELECT scholarship_name FROM scholarships WHERE scholarship_status != 'Expired'"
            params = []

            if intents.get("target_funding"):
                base_query += " AND funding_category = ?"
                params.append(intents["target_funding"])

            if not intents.get("bypass_filters"):
                level_map = {
                    "Bachelor": ["Undergraduates", "Unspecified"],
                    "Master": ["Graduates", "Unspecified"],
                    "PhD": ["Doctoral/PhD", "Unspecified"],
                    "Postdoc": ["Doctoral/PhD", "Unspecified"]
                }
                db_levels = level_map.get(profile.academic_level, ["Unspecified"])
                level_placeholders = ", ".join(["?"] * len(db_levels))

                base_query += f"""
                    AND (
                        eligible_nationality ILIKE ?
                        OR eligible_nationality ILIKE '%International%'
                        OR eligible_nationality ILIKE '%Unspecified%'
                        OR eligible_nationality ILIKE '%All%'
                        OR eligible_nationality ILIKE '%World%'
                    )
                    AND academic_level IN ({level_placeholders})
                """
                params.extend([f"%{profile.nationality}%"] + db_levels)

            results = conn.execute(base_query, params).fetchall()
            return [row[0] for row in results]
        finally:
            conn.close()

    def _formulate_query(self, profile: StudentProfile, base_query: str = "") -> str:
        """
        Constructs an enriched query string by combining the user's natural language input
        with structural data from their profile. Injects academic major, research interests,
        and technical skills as latent context to guide the dense vector search.

        Args:
            profile (StudentProfile): The active user profile.
            base_query (str, optional): The original user query. Defaults to "".

        Returns:
            str: The enriched query optimized for semantic embedding.
        """
        parts = []
        if base_query and base_query.strip():
            parts.append(base_query.strip())

        context_parts = [f"Academic Focus: {profile.academic_major}"]

        if profile.research_interests:
            context_parts.append(f"Research: {profile.research_interests}")

        if profile.skills:
            skills_str = ", ".join(profile.skills)
            context_parts.append(f"Skills: {skills_str}")

        parts.append("(" + " | ".join(context_parts) + ")")

        return " ".join(parts)

    def _rrf_fusion(self, dense_docs: List[Document], sparse_docs: List[Document], k: int = 60) -> List[Document]:
        """
        Implements Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score = 1 / (k + rank)
        """
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Process Dense results
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content  # Using content as unique ID
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc
            
        # Process Sparse results
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.page_content
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc
            
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[doc_id] for doc_id, score in sorted_docs]

    def match_scholarships(self, profile: StudentProfile, query: str = "", top_k: int = 5, fetch_k: int = 15) -> List[Any]:
        """
        Orchestrates the complete hybrid retrieval pipeline. Parses intents, applies
        deterministic SQL filters, executes dense and sparse searches, fuses results via RRF,
        re-ranks using a Cross-Encoder, enforces a strict confidence threshold, and finally
        enriches the successful chunks with heavy payload data (application process) from DuckDB.

        Args:
            profile (StudentProfile): The active user profile.
            query (str, optional): The user's search query. Defaults to "".
            top_k (int, optional): The final number of documents to return. Defaults to 5.
            fetch_k (int, optional): The number of documents to fetch during initial retrieval. Defaults to 15.

        Returns:
            List[Any]: The top-ranked, enriched Langchain documents passing all thresholds.
        """
        intents = self._detect_intents(query)

        valid_scholarships = self._deterministic_filter(profile, intents)
        if not valid_scholarships:
            return []

        search_query = self._formulate_query(profile, base_query=query)

        search_filter = {"scholarship_name": {"$in": valid_scholarships}}
        dense_results = self.vector_store.similarity_search(query=search_query, k=fetch_k, filter=search_filter)

        sparse_results = []
        if self.bm25_retriever:
            self.bm25_retriever.k = fetch_k * 3
            raw_sparse = self.bm25_retriever.invoke(search_query)
            sparse_results = [doc for doc in raw_sparse if doc.metadata.get('scholarship_name') in valid_scholarships][:fetch_k]

        hybrid_results = self._rrf_fusion(dense_results, sparse_results)
        hybrid_results = hybrid_results[:fetch_k]

        if not hybrid_results:
            return []

        pairs = [[search_query, doc.page_content] for doc in hybrid_results]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(hybrid_results, scores):
            doc.metadata['rerank_score'] = float(score)

        MIN_SCORE_THRESHOLD = 0.05
        valid_results = [doc for doc in hybrid_results if doc.metadata['rerank_score'] >= MIN_SCORE_THRESHOLD]

        valid_results.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
        final_results = valid_results[:top_k]

        if not final_results:
            logging.warning("System retrieved candidates, but ALL failed the %.3f confidence threshold.", MIN_SCORE_THRESHOLD)
            return []

        conn = duckdb.connect(self.duckdb_path)
        try:
            for doc in final_results:
                s_name = doc.metadata.get('scholarship_name')
                app_query = "SELECT application_process FROM scholarships WHERE scholarship_name = ?"
                result = conn.execute(app_query, [s_name]).fetchone()

                app_process = result[0] if result and result[0] else "Process not specified."
                doc.page_content += f"\n\n## Application Process\n{app_process}"
        finally:
            conn.close()

        return final_results



    # def match_scholarships(self, profile: StudentProfile, top_k: int = 5, fetch_k: int = 15) -> List[Any]:
    #     valid_scholarships = self._deterministic_filter(profile)
    #     if not valid_scholarships:
    #         return []
            
    #     search_query = self._formulate_query(profile)
        
    #     # --- 1. Dense Search (ChromaDB) ---
    #     search_filter = {"scholarship_name": {"$in": valid_scholarships}}
    #     dense_results = self.vector_store.similarity_search(query=search_query, k=fetch_k, filter=search_filter)
        
    #     # --- 2. Sparse Search (BM25) ---
    #     sparse_results = []
    #     if self.bm25_retriever:
    #         self.bm25_retriever.k = fetch_k * 3 # Fetch a wider net
    #         raw_sparse = self.bm25_retriever.invoke(search_query)
    #         # Post-filter BM25 results manually to respect DuckDB constraints
    #         sparse_results = [doc for doc in raw_sparse if doc.metadata.get('scholarship_name') in valid_scholarships][:fetch_k]
            
    #     # --- 3. RRF Fusion ---
    #     hybrid_results = self._rrf_fusion(dense_results, sparse_results)
    #     # Limit to fetch_k for the Cross-Encoder to maintain speed
    #     hybrid_results = hybrid_results[:fetch_k]
        
    #     if not hybrid_results:
    #         return []

    #     # --- 4. Cross-Encoder Re-ranking & Confidence Threshold ---
    #     pairs = [[search_query, doc.page_content] for doc in hybrid_results]
    #     scores = self.reranker.predict(pairs)
        
    #     for doc, score in zip(hybrid_results, scores):
    #         doc.metadata['rerank_score'] = float(score)
            
    #     # The Confidence Threshold implementation
    #     MIN_SCORE_THRESHOLD = 0.01
    #     valid_results = [doc for doc in hybrid_results if doc.metadata['rerank_score'] >= MIN_SCORE_THRESHOLD]
        
    #     valid_results.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
    #     final_results = valid_results[:top_k]
        
    #     if not final_results:
    #         logging.warning("System retrieved candidates, but ALL failed the %.3f confidence threshold.", MIN_SCORE_THRESHOLD)
            
    #     return final_results