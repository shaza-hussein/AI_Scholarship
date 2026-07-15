import re
import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ScholarshipDataProcessor:
    """
    A processor class for extracting and standardizing scholarship features 
    using a Cascade approach (Rule-based NER -> Zero-Shot Classification).
    """

    def __init__(self, use_zero_shot=False):
        """
        Initializes the NLP models and taxonomy dictionaries.
        
        Args:
            use_zero_shot (bool): Flag to enable/disable the heavy zero-shot fallback model.
        """
        logging.info("Initializing NLP Models...")
        # Load lightweight spaCy model (must be installed: python -m spacy download en_core_web_sm)
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Enable Zero-shot model if requested (consumes higher resources)
        self.use_zero_shot = use_zero_shot
        if self.use_zero_shot:
            logging.info("Loading Zero-Shot Classifier...")
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Taxonomy Standardization
        self.level_taxonomy = {
            "Undergraduates": ["undergraduate", "bachelor", "high school", "freshman", "bsc", "b.a", "b.s"],
            "Graduates": ["graduate", "master", "msc", "m.a", "mba", "postgraduate"],
            "Doctoral/PhD": ["phd", "doctoral", "doctorate", "post-doc", "postdoctoral"]
        }
        
        self.funding_taxonomy = {
            "Fully Funded": ["full ride", "fully funded", "covers tuition and living", "full tuition", "100% tuition"],
            "Partially Funded": ["partial", "covers tuition only", "discount", "contribution", "stipend only"]
        }
        
        self._build_spacy_rules()

    def _build_spacy_rules(self):
        """Builds Rule-based NER patterns for spaCy PhraseMatcher."""
        for standard_level, terms in self.level_taxonomy.items():
            patterns = [self.nlp.make_doc(text) for text in terms]
            self.matcher.add(f"LEVEL_{standard_level}", patterns)

    def expand_context(self, row):
        """
        Expands search scope: Merges descriptive text fields into one block for processing.
        """
        context_parts = [
            str(row.get('scholarship_name', '')),
            str(row.get('description', '')),
            str(row.get('eligibility', '')),
            str(row.get('scholarship_details', ''))
        ]
        return " ".join([p for p in context_parts if p.lower() not in ['nan', 'n/a', 'none']])

    def process_funding(self, row):
        """
        Processes the funding column to extract category and monetary value if present.
        """
        funding_raw = str(row.get('funding_type', '')).lower()
        context = self.expand_context(row).lower()
        
        result = {
            'funding_category': 'Variable / Unspecified',
            'funding_amount': None
        }

        # 1. Regex to extract monetary values
        money_pattern = r'(\$|€|£)\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+[kK])'
        match = re.search(money_pattern, funding_raw + " " + context)
        
        if match:
            result['funding_amount'] = "".join(match.groups())
            result['funding_category'] = 'Fixed Grant'
            return pd.Series(result)

        # 2. Dictionary Matching if no number is found
        for standard_cat, keywords in self.funding_taxonomy.items():
            if any(kw in funding_raw for kw in keywords) or any(kw in context for kw in keywords):
                result['funding_category'] = standard_cat
                return pd.Series(result)

        return pd.Series(result)

    def process_degree_level(self, row):
        """
        Processes academic levels using a Cascade approach.
        """
        raw_degree = str(row.get('degree_level', ''))
        context = self.expand_context(row)
        
        result = {
            'academic_major': 'All Disciplines', # Default value
            'academic_level': 'Unspecified'      # Default value
        }

        # Proactive processing: if the original field contains a major, relocate it
        known_majors = ['Computer Science', 'Cybersecurity', 'Theology/Ministry', 'Web Design / Creative Arts']
        if any(major in raw_degree for major in known_majors):
            result['academic_major'] = raw_degree

        # --- CASCADE STAGE 1: Rule-Based NER (spaCy) ---
        doc = self.nlp(context)
        matches = self.matcher(doc)
        
        if matches:
            # Take the first identified standard category
            match_id, start, end = matches[0]
            rule_id = self.nlp.vocab.strings[match_id]
            result['academic_level'] = rule_id.replace("LEVEL_", "")
            return pd.Series(result)

        # --- CASCADE STAGE 2: Zero-Shot Classification (Fallback) ---
        if self.use_zero_shot and result['academic_level'] == 'Unspecified':
            candidate_labels = list(self.level_taxonomy.keys())
            try:
                # Pass only the first 1000 characters to prevent memory overflow
                clf_result = self.classifier(context[:1000], candidate_labels)
                # If model confidence is above 50%
                if clf_result['scores'][0] > 0.5:
                    result['academic_level'] = clf_result['labels'][0]
            except Exception as e:
                logging.warning(f"Zero-shot failed: {e}")

        return pd.Series(result)

# ==========================================
# Pipeline Usage Example
# ==========================================
if __name__ == "__main__":
    # Simulated raw data
    data = [{
        'degree_level': 'Computer Science', 
        'funding_type': 'Will receive $10,000 annually',
        'description': 'This scholarship is for outstanding bachelor students in tech.',
        'eligibility': 'Must maintain a 3.5 GPA.'
    }]
    df = pd.DataFrame(data)
    
    # Initialize processor
    processor = ScholarshipDataProcessor(use_zero_shot=False)
    
    # Apply processing to new columns
    logging.info("Processing Funding...")
    df[['funding_category', 'funding_amount']] = df.apply(processor.process_funding, axis=1)
    
    logging.info("Processing Degree Levels...")
    df[['academic_major', 'academic_level']] = df.apply(processor.process_degree_level, axis=1)
    
    # Display results
    print("\n--- Processed Data ---")
    print(df[['academic_major', 'academic_level', 'funding_category', 'funding_amount']].to_string())