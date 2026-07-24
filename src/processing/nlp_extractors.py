import os
import re
import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline
import pandas as pd
import logging
import dateutil.parser as dparser
import ollama

import json
import dateutil.parser as dparser
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

load_dotenv()

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

        # Initialize Groq LLM Client for advanced extraction(deadline) 
        self.api_key = os.environ.get("GROQ_API_KEY")
        if self.api_key:
            logging.info("Initializing LLM Client...")
            self.llm_client = Groq(api_key=self.api_key)
            self.llm_model = "llama-3.3-70b-versatile"
            # self.llm_model = "llama-3.1-8b-instant"
        else:
            logging.warning("API key missing. LLM extraction disabled.")
            self.llm_client = None

        # Taxonomy Standardization
        self.level_taxonomy = {
            "Undergraduates": ["undergraduate", "bachelor", "high school", "freshman", "bsc", "b.a", "b.s"],
            "Graduates": ["graduate", "master", "msc", "m.a", "mba", "postgraduate"],
            "Doctoral/PhD": ["phd", "doctoral", "doctorate", "post-doc", "postdoctoral"]
        }
        

        self.funding_taxonomy = {
            "Fully Funded": [
                "full ride", "fully funded", "covers tuition and living", "full tuition", 
                "monthly payment", "monthly payments", "monthly stipend", "stipend", 
                "living expenses", "travel allowance", "health insurance", "comprehensive"
            ],
            "Partially Funded": [
                "partial", "covers tuition only", "discount", "contribution", 
                "stipend only", "one-time", "lump sum", "fee reduction"
            ]
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
        
        
        boilerplate = "fully/partially funded (check details)"
        if boilerplate in funding_raw:
            funding_raw = funding_raw.replace(boilerplate, "").strip()
            
        context = self.expand_context(row).lower()
        combined_text = funding_raw + " " + context
        
        result = {
            'funding_category': 'Variable / Unspecified',
            'funding_amount': None
        }

        # 2. An advanced regular expression that captures currencies whether they appear before the number (e.g., €900) or after it (e.g., 900 euros).
        # Supports: $, €, £, eur, usd, euro, euros
        money_pattern = r'((?:\$|€|£|eur|usd)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:€|eur|euros?|usd|\$))'
        match = re.search(money_pattern, combined_text, re.IGNORECASE)
        
        if match:
            # Extract the amount(e.g., "934 euros" or "€1200")
            result['funding_amount'] = match.group(1).strip()
            result['funding_category'] = 'Fixed Grant'
            return pd.Series(result)

        # 3. Expanded Dictionary Matching
        for standard_cat, keywords in self.funding_taxonomy.items():
            if any(kw in combined_text for kw in keywords):
                result['funding_category'] = standard_cat
                return pd.Series(result)

        return pd.Series(result)


# updated process_degree_level method with a more robust cascade approach, including explicit tag checks, rule-based NER, and optional zero-shot classification for fallback.
    def process_degree_level(self, row):
        raw_degree = str(row.get('degree_level', ''))
        context = self.expand_context(row)
        
        tags = row.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip(" '\"[]") for t in tags.split(',')]
        elif not isinstance(tags, list):
            tags = []
            
        tags_lower = [str(t).lower() for t in tags]
        
        result = {'academic_level': 'Unspecified'}

        tag_mapping = {
            'undergraduates': 'Undergraduates',
            'graduates': 'Graduates',
            'doctoral candidates/phd students': 'Doctoral/PhD',
            'doctoral/phd': 'Doctoral/PhD',
            'postdoctoral researchers': 'Doctoral/PhD'
        }
        
        for tag in tags_lower:
            if tag in tag_mapping:
                result['academic_level'] = tag_mapping[tag]
                return pd.Series(result)

        doc = self.nlp(context)
        matches = self.matcher(doc)
        
        if matches:
            match_id, start, end = matches[0]
            rule_id = self.nlp.vocab.strings[match_id]
            result['academic_level'] = rule_id.replace("LEVEL_", "")
            return pd.Series(result)

        if self.use_zero_shot and result['academic_level'] == 'Unspecified':
            candidate_labels = list(self.level_taxonomy.keys())
            try:
                clf_result = self.classifier(context[:1000], candidate_labels)
                if clf_result['scores'][0] > 0.5:
                    result['academic_level'] = clf_result['labels'][0]
            except Exception as e:
                logging.warning(f"Zero-shot failed: {e}")

        return pd.Series(result)
    

    def process_academic_major(self, row):
        import json
        import ollama
        
        raw_degree = str(row.get('degree_level', '')).strip()
        context = self.expand_context(row)
        
        combined_text = f"Raw Value: {raw_degree} | Details: {context[:4000]}"
        
        prompt = f"""
        Extract the academic major or field of study required for this scholarship.
        
        RULES:
        1. If the scholarship targets specific fields (e.g., Computer Science, Engineering, Medicine), extract them as a comma-separated string.
        2. If the text explicitly states it is open to all subjects/fields, return "All Disciplines".
        3. If no specific field is mentioned at all, return "Unspecified".
        4. Do NOT include degree levels like Bachelor, Master, or PhD in the output.
        
        Output STRICTLY in valid JSON format: {{"academic_major": "extracted fields"}}
        
        Text to process:
        {combined_text}
        """
        
        try:
            response = ollama.chat(
                model='qwen2.5:7b',
                messages=[
                    {"role": "system", "content": "You output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0}
            )
            
            content = response['message']['content']
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed_json = json.loads(content.strip())
            major = parsed_json.get("academic_major", "Unspecified")
            
            return pd.Series({'academic_major': major})
            
        except Exception as e:
            logging.error(f"Major Extraction Error: {e}")
            return pd.Series({'academic_major': 'Unspecified'})




    # process country
    def process_country(self, row):
        """
        Extracts host country and eligible nationality using explicit tags 
        as the primary source of truth, falling back to NLP for unstructured data.
        """
        raw_country = str(row.get('country', '')).strip().lower()
        context = self.expand_context(row)
        
        tags = row.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip(" '\"[]") for t in tags.split(',')]
        elif not isinstance(tags, list):
            tags = []
            
        tags_lower = [str(t).lower() for t in tags]
        
        result = {
            'host_country': 'Unspecified',
            'eligible_nationality': 'International / All'
        }

        # HIERARCHY LEVEL 1: Check Tags for exact Host Country
        if 'germany' in tags_lower or 'daad' in tags_lower:
            result['host_country'] = 'Germany'
        elif 'college selection' in tags_lower or 'usa' in tags_lower or raw_country == 'usa':
            result['host_country'] = 'USA'

        # HIERARCHY LEVEL 1: Check Tags for Target Nationality
        target_nationalities = ['egypt', 'syria', 'jordan', 'saudi arabia']
        found_targets = set()
        
        for tn in target_nationalities:
            if tn in tags_lower:
                found_targets.add(tn.title())
                
        if found_targets:
            result['eligible_nationality'] = ", ".join(sorted(list(found_targets)))
            
        # If both fields are filled via tags, bypass NLP entirely (Performance boost)
        if result['host_country'] != 'Unspecified' and result['eligible_nationality'] != 'International / All':
            return pd.Series(result)

        # HIERARCHY LEVEL 2: NLP Contextual Extraction (Fallback for missing fields)
        host_keywords = ['study in', 'travel to', 'universities in', 'located in', 'host', 'tenable in']
        nat_keywords = ['citizens of', 'citizenship', 'open to', 'nationals of', 'from', 'passport']
        
        nlp_hosts = set()
        nlp_nats = set()
        
        doc = self.nlp(context)
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                start_token = max(0, ent.start - 5)
                preceding_text = doc[start_token:ent.start].text.lower()
                
                is_host = any(kw in preceding_text for kw in host_keywords)
                is_nat = any(kw in preceding_text for kw in nat_keywords)
                
                if is_host and result['host_country'] == 'Unspecified':
                    nlp_hosts.add(ent.text)
                elif is_nat and not found_targets:
                    nlp_nats.add(ent.text)

        if nlp_hosts and result['host_country'] == 'Unspecified':
            result['host_country'] = ", ".join(sorted(list(nlp_hosts)))
            
        if nlp_nats and result['eligible_nationality'] == 'International / All':
            result['eligible_nationality'] = ", ".join(sorted(list(nlp_nats)))

        return pd.Series(result)
    

    # deadline processing with hybrid approach: rule-based parsing + LLM fallback
    def parse_exact_date(self, text):
        try:
            dt = dparser.parse(text, fuzzy=False)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return None

    def extract_via_llm(self, context_text):
        """
        Extracts the application deadline strictly in YYYY-MM-DD format
        using a local LLM via Ollama to ensure infinite free scalability.
        """
        
        
        prompt = f"""
        Extract the application submission deadline from the text below.
        
        CRITICAL RULES:
        1. Find ONLY the application submission deadline.
        2. IGNORE all other dates (e.g., funding starts, selection decisions, program start dates, research stays).
        3. You must format the output exactly as YYYY-MM-DD.
        4. If there is no specific application deadline mentioned, you must output "Not Specified".
         
        Output strictly in valid JSON format: {{"deadline": "YYYY-MM-DD"}} or {{"deadline": "Not Specified"}}
        
        Text to process:
        {context_text}
        """
        
        try:
            response = ollama.chat(
                model='qwen2.5:7b',
                messages=[
                    {"role": "system", "content": "You output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0}
            )
            
            content = response['message']['content']
            
            # Clean up potential markdown formatting in local LLM outputs
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            parsed_json = json.loads(content.strip())
            return parsed_json.get("deadline", "Not Specified")
        except Exception as e:
            logging.error(f"Local LLM Error: {e}")
            return "Not Specified"


    def process_deadline(self, row):
        raw_deadline = str(row.get('deadline', '')).strip()
        result = {'standardized_deadline': 'Not Specified'}
        
        useless_vals = ['not specified', 'varies', 'nan', 'none', 'rolling', 'continuous']
        is_useless = raw_deadline.lower() in useless_vals
        
       
        if not is_useless and len(raw_deadline) < 30:
            parsed = self.parse_exact_date(raw_deadline)
            if parsed:
                result['standardized_deadline'] = parsed
                return pd.Series(result)

       
        context = self.expand_context(row)
        

        if is_useless:
            combined_text = f"Scholarship Details: {context}"
        else:
            combined_text = f"Extracted Text: {raw_deadline} | Scholarship Details: {context}"
        

        llm_result = self.extract_via_llm(combined_text[:8000])
        result['standardized_deadline'] = llm_result
        
        return pd.Series(result)

