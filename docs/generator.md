# AI Generation & Interactive Layer

This module transforms raw retrieved documents into an intelligent, conversational, and multilingual response. It employs a "Late Enrichment" pattern to fetch application steps from DuckDB just-in-time, uses intent-driven routing, and leverages Groq's LPU infrastructure (Llama-3) to provide accurate, hallucination-free academic assistance.

::: src.rag.generator