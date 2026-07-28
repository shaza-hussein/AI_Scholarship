# Dual-Store RAG Pipeline

This module orchestrates the ingestion of cleaned scholarship data into the dual-store AI architecture. It handles the simultaneous creation of the DuckDB analytical database for deterministic filtering and the ChromaDB semantic vector store for hybrid retrieval.

::: src.pipeline.rag_pipeline