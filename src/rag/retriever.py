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

    def _deterministic_filter(self, profile: StudentProfile) -> List[str]:
        conn = duckdb.connect(self.duckdb_path)
        try:
            level_map = {
                "Bachelor": ["Undergraduates", "Unspecified"],
                "Master": ["Graduates", "Unspecified"],
                "PhD": ["Doctoral/PhD", "Unspecified"],
                "Postdoc": ["Doctoral/PhD", "Unspecified"] 
            }
            db_levels = level_map.get(profile.academic_level, ["Unspecified"])
            level_placeholders = ", ".join(["?"] * len(db_levels))
            
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
            params = [f"%{profile.nationality}%"] + db_levels
            results = conn.execute(query, params).fetchall()
            valid_names = [row[0] for row in results]
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

    def match_scholarships(self, profile: StudentProfile, top_k: int = 5, fetch_k: int = 15) -> List[Any]:
        valid_scholarships = self._deterministic_filter(profile)
        if not valid_scholarships:
            return []
            
        search_query = self._formulate_query(profile)
        
        # --- 1. Dense Search (ChromaDB) ---
        search_filter = {"scholarship_name": {"$in": valid_scholarships}}
        dense_results = self.vector_store.similarity_search(query=search_query, k=fetch_k, filter=search_filter)
        
        # --- 2. Sparse Search (BM25) ---
        sparse_results = []
        if self.bm25_retriever:
            self.bm25_retriever.k = fetch_k * 3 # Fetch a wider net
            raw_sparse = self.bm25_retriever.invoke(search_query)
            # Post-filter BM25 results manually to respect DuckDB constraints
            sparse_results = [doc for doc in raw_sparse if doc.metadata.get('scholarship_name') in valid_scholarships][:fetch_k]
            
        # --- 3. RRF Fusion ---
        hybrid_results = self._rrf_fusion(dense_results, sparse_results)
        # Limit to fetch_k for the Cross-Encoder to maintain speed
        hybrid_results = hybrid_results[:fetch_k]
        
        if not hybrid_results:
            return []

        # --- 4. Cross-Encoder Re-ranking & Confidence Threshold ---
        pairs = [[search_query, doc.page_content] for doc in hybrid_results]
        scores = self.reranker.predict(pairs)
        
        for doc, score in zip(hybrid_results, scores):
            doc.metadata['rerank_score'] = float(score)
            
        # The Confidence Threshold implementation
        MIN_SCORE_THRESHOLD = 0.01
        valid_results = [doc for doc in hybrid_results if doc.metadata['rerank_score'] >= MIN_SCORE_THRESHOLD]
        
        valid_results.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
        final_results = valid_results[:top_k]
        
        if not final_results:
            logging.warning("System retrieved candidates, but ALL failed the %.3f confidence threshold.", MIN_SCORE_THRESHOLD)
            
        return final_results