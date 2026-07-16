import os
import sys
import glob
import logging
import pandas as pd
from tqdm import tqdm

# Add project root directory to sys.path to ensure correct module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.processing.nlp_extractors import ScholarshipDataProcessor

# Configure logging for production tracing
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    logging.info("Starting the Scholarship ETL Pipeline...")

    # Define directory paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data_Json')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Create output directory if it does not exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Load raw data
    logging.info(f"Loading raw JSON files from: {RAW_DATA_DIR}")
    file_paths = glob.glob(os.path.join(RAW_DATA_DIR, '*_scholarships_data*.json'))
    
    if not file_paths:
        logging.error("No JSON files found in the raw data directory. Exiting.")
        return

    all_data = []
    for file_path in file_paths:
        try:
            df_temp = pd.read_json(file_path)
            df_temp['source_file'] = os.path.basename(file_path)
            all_data.append(df_temp)
            logging.info(f"Loaded: {os.path.basename(file_path)} ({len(df_temp)} records)")
        except Exception as e:
            logging.error(f"Error loading {os.path.basename(file_path)}: {e}")

    df_raw = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total records aggregated: {len(df_raw)}")

    # Initialize NLP Processor
    logging.info("Initializing the NLP Processor...")
    processor = ScholarshipDataProcessor(use_zero_shot=False)

    # Apply processing functions
    tqdm.pandas(desc="Processing Funding")
    logging.info("Extracting Funding Categories and Amounts...")
    df_raw[['funding_category', 'funding_amount']] = df_raw.progress_apply(processor.process_funding, axis=1)

    tqdm.pandas(desc="Processing Degrees")
    logging.info("Extracting Academic Levels and Majors...")
    df_raw[['academic_major', 'academic_level']] = df_raw.progress_apply(processor.process_degree_level, axis=1)

    tqdm.pandas(desc="Processing Countries")
    logging.info("Extracting Host Countries and Eligible Nationalities...")
    df_raw[['host_country', 'eligible_nationality']] = df_raw.progress_apply(processor.process_country, axis=1)

    # Save processed data
    output_file_json = os.path.join(PROCESSED_DATA_DIR, 'master_scholarships_clean.json')
    
    logging.info("Saving processed data...")
    df_raw.to_json(output_file_json, orient='records', indent=4)
    
    logging.info(f"Pipeline completed successfully. Clean data saved to: {output_file_json}")

if __name__ == "__main__":
    main()