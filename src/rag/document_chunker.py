import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class HybridDocumentChunker:
    """
    Advanced context-aware document chunking architecture for RAG systems.
    
    This class executes a two-stage hybrid splitting strategy:
    1. Semantic Splitting: Utilizes MarkdownHeaderTextSplitter to divide documents by 
       logical sections and injects these headers into the metadata.
    2. Token-Safe Splitting: Applies RecursiveCharacterTextSplitter on oversized sections 
       to ensure strict compliance with embedding model context windows.
       
    Architectural Upgrades (Dual-Store Readiness):
    - Ghost Chunk Pruning: Automatically discards chunks with insignificant semantic weight.
    - Context Bloat Prevention: Slims down metadata by dropping heavy payload fields 
      (e.g., application_process) to preserve the LLM context window.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_length: int = 30) -> None:
        """
        Initializes the hybrid splitting engines with optimized token thresholds.
        
        Args:
            chunk_size (int): The maximum character limit per chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
            min_chunk_length (int): Minimum length for a chunk to be considered semantically valid.
        """
        self.min_chunk_length = min_chunk_length
        
        self.headers_to_split_on = [
            ("#", "Document_Title"),
            ("##", "Section_Title")
        ]
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _flatten_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extracts, flattens, and slims the nested dictionary for Vector DB compliance.
        
        Drops 'application_process' to prevent LLM context overflow during RAG retrieval,
        adhering to the Dual-Store architecture where heavy metadata is kept in analytical DBs.
        
        Args:
            row (pd.Series): A single row containing the 'rag_metadata' column.
            
        Returns:
            Dict[str, Any]: A flat, optimized dictionary compliant with Vector DB schemas.
        """
        metadata: Dict[str, Any] = row.get('rag_metadata', {})
        if not isinstance(metadata, dict):
            return {"scholarship_name": row.get('scholarship_name', 'Unknown')}
            
        if 'application_process' in metadata:
            del metadata['application_process']
            
        return metadata

    def load_and_filter(self, json_path: str) -> List[Document]:
        """
        Loads the JSON file and converts the tabular structure into LangChain Documents.
        
        Args:
            json_path (str): The path to the processed JSON data.
            
        Returns:
            List[Document]: Base documents containing the un-chunked text and slimmed metadata.
        """
        df = pd.read_json(json_path)
        
        required_columns = ['scholarship_name', 'rag_document', 'rag_metadata']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CRITICAL ERROR: Missing required column '{col}' in the dataset.")
                
        base_documents: List[Document] = []
        
        for _, row in df.iterrows():
            text_content = str(row['rag_document'])
            if not text_content.strip():
                continue
                
            slim_meta = self._flatten_metadata(row)
            
            doc = Document(page_content=text_content, metadata=slim_meta)
            base_documents.append(doc)
            
        return base_documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Executes the hybrid chunking pipeline and prunes semantic noise.
        
        Args:
            documents (List[Document]): The un-chunked source documents.
            
        Returns:
            List[Document]: The final, embedding-ready chunks without ghost segments.
        """
        final_chunks: List[Document] = []
        
        for doc in documents:
            markdown_chunks = self.markdown_splitter.split_text(doc.page_content)
            
            for md_chunk in markdown_chunks:
                merged_metadata = {**doc.metadata, **md_chunk.metadata}
                md_chunk.metadata = merged_metadata
                
            recursive_chunks = self.recursive_splitter.split_documents(markdown_chunks)
            
            valid_chunks = [
                chunk for chunk in recursive_chunks 
                if len(chunk.page_content.strip()) >= self.min_chunk_length
            ]
            final_chunks.extend(valid_chunks)
            
        return final_chunks