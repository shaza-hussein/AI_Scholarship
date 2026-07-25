"""
Scholarship ETL (Extract, Transform, Load) Pipeline Orchestrator.

This module serves as the central orchestration layer for the data preparation pipeline.
It transitions raw, unstructured scraped data into a highly structured, vector-ready format 
suitable for Retrieval-Augmented Generation (RAG) systems. 

Architectural Phases:
1. Ingestion: Aggregates fragmented JSON files from data scrapers, maintaining data lineage.
2. NLP/LLM Extraction: Leverages the ScholarshipDataProcessor to extract and standardize 
   critical entities (e.g., funding amounts, deadlines, academic majors).
3. Fallback Imputation: Applies heuristic fallbacks to missing navigational fields 
   (e.g., application processes) to ensure the LLM always has actionable context.
4. RAG Serialization: Utilizes RagDocumentBuilder to generate semantically clean 
   documents and strictly typed metadata payloads for vector database indexing.
"""

import os
import sys
import glob
import logging
import pandas as pd
from tqdm import tqdm
from typing import List

# ---------------------------------------------------------
# Path Configuration & Environment Setup
# ---------------------------------------------------------
# Append the project root directory to sys.path. 
# This ensures deterministic module resolution regardless of the execution context.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.processing.nlp_extractors import ScholarshipDataProcessor
from src.processing.rag_formatter import RagDocumentBuilder

# ---------------------------------------------------------
# Observability & Tracing Configuration
# ---------------------------------------------------------
# Configure root logger for production-grade tracing.
# Standardizes output format to include timestamps and severity levels for precise debugging.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main() -> None:
    """
    Executes the main ETL pipeline workflow.
    
    The execution flow is heavily tracked using progress bars (tqdm) and logging 
    to monitor the performance of the underlying LLM/NLP components.
    """
    logging.info("Starting the Scholarship ETL Pipeline...")

    # 1. Define architectural directory paths dynamically based on the current file location
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data_Json')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data_Json', 'processed')
    
    # Ensure the output directory structure exists to prevent IOError during serialization
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Phase 1: Data Ingestion & Lineage Tracking
    # ---------------------------------------------------------
    logging.info(f"Loading raw JSON files from: {RAW_DATA_DIR}")
    file_paths: List[str] = glob.glob(os.path.join(RAW_DATA_DIR, '*_scholarships_data*.json'))
    
    if not file_paths:
        logging.error("No JSON files found in the raw data directory. Exiting.")
        return

    all_data = []
    for file_path in file_paths:
        try:
            df_temp = pd.read_json(file_path)
            # Inject a 'source_file' column to maintain data lineage and traceability
            df_temp['source_file'] = os.path.basename(file_path)
            all_data.append(df_temp)
            logging.info(f"Loaded: {os.path.basename(file_path)} ({len(df_temp)} records)")
        except Exception as e:
            logging.error(f"Error loading {os.path.basename(file_path)}: {e}")

    # Aggregate all fragmented dataframes into a single unified schema
    df_raw = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total records aggregated: {len(df_raw)}")

    # ---------------------------------------------------------
    # Phase 2: NLP & LLM Entity Extraction
    # ---------------------------------------------------------
    # Initialize the primary extraction engine. 
    # use_zero_shot=False implies reliance on local LLMs (e.g., Ollama) or deterministic rules.
    logging.info("Initializing the NLP Processor...")
    processor = ScholarshipDataProcessor(use_zero_shot=False)

    # Apply sequential extraction functions. 
    # Note: progress_apply is utilized to monitor long-running LLM inference tasks.

    # Extract structured financial criteria
    tqdm.pandas(desc="Processing Funding")
    logging.info("Extracting Funding Categories and Amounts...")
    df_raw[['funding_category', 'funding_amount']] = df_raw.progress_apply(processor.process_funding, axis=1)

    # Standardize degree requirements
    tqdm.pandas(desc="Processing Academic Levels")
    logging.info("Extracting Academic Levels...")
    df_raw['academic_level'] = df_raw.progress_apply(processor.process_degree_level, axis=1)['academic_level']

    # Leverage LLM for complex contextual classification of academic disciplines
    tqdm.pandas(desc="Processing Academic Majors")
    logging.info("Extracting Academic Majors via LLM...")
    df_raw['academic_major'] = df_raw.progress_apply(processor.process_academic_major, axis=1)['academic_major']

    # Normalize geopolitical entities
    tqdm.pandas(desc="Processing Countries")
    logging.info("Extracting Host Countries and Eligible Nationalities...")
    df_raw[['host_country', 'eligible_nationality']] = df_raw.progress_apply(processor.process_country, axis=1)

    # Standardize temporal data into unified date formats
    tqdm.pandas(desc="Processing Deadlines")
    logging.info("Extracting Deadlines via Hybrid NLP-LLM Engine...")
    df_raw['standardized_deadline'] = df_raw.progress_apply(processor.process_deadline, axis=1)['standardized_deadline']

    # ---------------------------------------------------------
    # Phase 3: Sanitization & Metadata Payload Imputation
    # ---------------------------------------------------------
    # The application process is critical for the RAG LLM to answer "How do I apply?".
    # This block identifies dead text or missing instructions and injects a deterministic fallback.
    tqdm.pandas(desc="Processing Application Process")
    logging.info("Cleaning textual columns and applying safe fallbacks...")
    df_raw['application_process'] = df_raw['application_process'].fillna('').astype(str).str.strip()
    
    useless_vals = ['nan', 'none', 'n/a', 'not specified', 'tba', '']
    fallback_message = "Please visit the official scholarship website for detailed application instructions and requirements."
    
    # Determine which records lack sufficient semantic instructions (less than 10 words or matching useless values)
    missing_condition = (
        df_raw['application_process'].str.lower().isin(useless_vals) | 
        (df_raw['application_process'].str.split().str.len() < 10)
    )
    df_raw.loc[missing_condition, 'application_process'] = fallback_message


    # ---------------------------------------------------------
    # Phase 4: RAG Architecture Formatting
    # ---------------------------------------------------------
    # Transforms the tabular pandas structure into vector-ready documents and strict metadata payload dictionaries.
    tqdm.pandas(desc="Processing Textual Columns")
    logging.info("Building Semantic Documents & Metadata for RAG...")
    
    rag_builder = RagDocumentBuilder()
    df_raw['rag_document'] = df_raw.apply(rag_builder.build_document, axis=1)
    df_raw['rag_metadata'] = df_raw.apply(rag_builder.build_metadata, axis=1)


    # ---------------------------------------------------------
    # Phase 5: Serialization
    # ---------------------------------------------------------
    output_file_json = os.path.join(PROCESSED_DATA_DIR, 'master_scholarships_clean.json')
    
    logging.info("Saving processed data...")
    df_raw.to_json(output_file_json, orient='records', indent=4)
    
    logging.info(f"Pipeline completed successfully. Clean data saved to: {output_file_json}")

if __name__ == "__main__":
    main()





# import os
# import sys
# import glob
# import logging
# import pandas as pd
# from tqdm import tqdm
# from src.processing.rag_formatter import RagDocumentBuilder

# # Add project root directory to sys.path to ensure correct module imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from src.processing.nlp_extractors import ScholarshipDataProcessor

# # Configure logging for production tracing
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# def main():
#     logging.info("Starting the Scholarship ETL Pipeline...")

#     # Define directory paths
#     BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
#     RAW_DATA_DIR = os.path.join(BASE_DIR, 'data_Json')
#     PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data_Json', 'processed')
    
#     # Create output directory if it does not exist
#     os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

#     # Load raw data
#     logging.info(f"Loading raw JSON files from: {RAW_DATA_DIR}")
#     file_paths = glob.glob(os.path.join(RAW_DATA_DIR, '*_scholarships_data*.json'))
    
#     if not file_paths:
#         logging.error("No JSON files found in the raw data directory. Exiting.")
#         return

#     all_data = []
#     for file_path in file_paths:
#         try:
#             df_temp = pd.read_json(file_path)
#             df_temp['source_file'] = os.path.basename(file_path)
#             all_data.append(df_temp)
#             logging.info(f"Loaded: {os.path.basename(file_path)} ({len(df_temp)} records)")
#         except Exception as e:
#             logging.error(f"Error loading {os.path.basename(file_path)}: {e}")

#     df_raw = pd.concat(all_data, ignore_index=True)
#     logging.info(f"Total records aggregated: {len(df_raw)}")

#     # Initialize NLP Processor
#     logging.info("Initializing the NLP Processor...")
#     processor = ScholarshipDataProcessor(use_zero_shot=False)

#     # Apply processing functions

#     # funding
#     tqdm.pandas(desc="Processing Funding")
#     logging.info("Extracting Funding Categories and Amounts...")
#     df_raw[['funding_category', 'funding_amount']] = df_raw.progress_apply(processor.process_funding, axis=1)

#     # degree_level
#     tqdm.pandas(desc="Processing Academic Levels")
#     logging.info("Extracting Academic Levels...")
#     df_raw['academic_level'] = df_raw.progress_apply(processor.process_degree_level, axis=1)['academic_level']

#     tqdm.pandas(desc="Processing Academic Majors")
#     logging.info("Extracting Academic Majors via LLM...")
#     df_raw['academic_major'] = df_raw.progress_apply(processor.process_academic_major, axis=1)['academic_major']

#     # country
#     tqdm.pandas(desc="Processing Countries")
#     logging.info("Extracting Host Countries and Eligible Nationalities...")
#     df_raw[['host_country', 'eligible_nationality']] = df_raw.progress_apply(processor.process_country, axis=1)

#     # deadline
#     tqdm.pandas(desc="Processing Deadlines")
#     logging.info("Extracting Deadlines via Hybrid NLP-LLM Engine...")
#     df_raw['standardized_deadline'] = df_raw.progress_apply(processor.process_deadline, axis=1)['standardized_deadline']

#     # application_process
#     tqdm.pandas(desc="Processing Application Process")
#     logging.info("Cleaning textual columns and applying safe fallbacks...")
#     df_raw['application_process'] = df_raw['application_process'].fillna('').astype(str).str.strip()
    
#     useless_vals = ['nan', 'none', 'n/a', 'not specified', 'tba', '']
#     fallback_message = "Please visit the official scholarship website for detailed application instructions and requirements."
    
#     missing_condition = (
#         df_raw['application_process'].str.lower().isin(useless_vals) | 
#         (df_raw['application_process'].str.split().str.len() < 10)
#     )
#     df_raw.loc[missing_condition, 'application_process'] = fallback_message


#     # RAG Document Formatting
#     tqdm.pandas(desc="Processing Textual Columns")
#     logging.info("Building Semantic Documents & Metadata for RAG...")
    
#     rag_builder = RagDocumentBuilder()
#     df_raw['rag_document'] = df_raw.apply(rag_builder.build_document, axis=1)
#     df_raw['rag_metadata'] = df_raw.apply(rag_builder.build_metadata, axis=1)


#     # Save processed data
#     output_file_json = os.path.join(PROCESSED_DATA_DIR, 'master_scholarships_clean.json')
    
#     logging.info("Saving processed data...")
#     df_raw.to_json(output_file_json, orient='records', indent=4)
    
#     logging.info(f"Pipeline completed successfully. Clean data saved to: {output_file_json}")

# if __name__ == "__main__":
#     main()