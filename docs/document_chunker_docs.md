# Hybrid Document Chunking Architecture

This module implements the LangChain chunking strategy. It solves the context-loss problem by utilizing a hybrid approach (Markdown splitting followed by recursive character splitting) and ensures ChromaDB metadata compliance through flattening.

::: rag.document_chunker.HybridDocumentChunker