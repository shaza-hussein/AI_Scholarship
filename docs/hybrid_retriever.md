# Advanced Hybrid Retriever

This module implements a multi-stage retrieval architecture designed for maximum precision and zero hallucination. It orchestrates deterministic SQL filtering (DuckDB), semantic vector search (ChromaDB), in-memory lexical keyword matching (BM25), and cross-encoder re-ranking to deliver the most relevant scholarship candidates.

::: src.rag.retriever