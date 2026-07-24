import re
import pandas as pd
import numpy as np

class RagDocumentBuilder:
    def __init__(self):
        self.html_pattern = re.compile(r'<[^>]+>')
        self.separator_pattern = re.compile(r'-{3,}|={3,}|\*{3,}')
        self.bullet_pattern = re.compile(r'[•-]\s*\n')
        self.newline_pattern = re.compile(r'\n{3,}')
        # self.boilerplate_pattern = re.compile(
        #     r'(?i)\b(click here|read more|all rights reserved|subscribe(?: to)?|download(?: pdf)?|official website|apply now|learn more|contact us|faq(?:s)?|frequently asked questions|privacy policy|terms (?:of use|and conditions)|cookie policy|copyright|follow us(?: on)?|share this|newsletter)\b'
        # )
        # 2. Boilerplate Pattern (Extended)
        self.boilerplate_pattern = re.compile(
            r'(?i)\b('
            r'click here|read more|all rights reserved|subscribe(?: to)?|download(?: pdf)?|official website|'
            r'apply now|learn more|contact us|faq(?:s)?|frequently asked questions|'
            r'privacy policy|terms (?:of use|and conditions)|cookie policy|copyright|'
            r'follow us(?: on)?|share this|newsletter'
            r')\b'
        )
        
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')

    def clean_text(self, text, min_words=10):
        if pd.isna(text) or str(text).strip().lower() in ['nan', 'none', 'n/a', '']:
            return ""
            
        text = str(text)
        text = self.html_pattern.sub(' ', text)
        text = self.separator_pattern.sub(' ', text)
        text = self.bullet_pattern.sub('- ', text)
        text = self.boilerplate_pattern.sub(' ', text)
        text = self.url_pattern.sub('[LINK]', text)
        text = self.email_pattern.sub('[EMAIL]', text)
        text = self.newline_pattern.sub('\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        cleaned_text = text.strip()
        
        if len(cleaned_text.split()) < min_words:
            return ""
            
        return cleaned_text

    def _safe_meta(self, val, default="Unspecified"):
        if pd.isna(val) or val is None or str(val).strip().lower() in ['nan', 'none', 'n/a', '']:
            return default
        return str(val).strip()

    def build_document(self, row):
        doc_parts = []
        scholarship_name = self._safe_meta(row.get('scholarship_name', 'Unnamed Scholarship'))
        doc_parts.append(f"# Scholarship: {scholarship_name}")
        
        desc = self.clean_text(row.get('description', ''))
        details = self.clean_text(row.get('scholarship_details', ''))
        eligibility = self.clean_text(row.get('eligibility', ''))
        
        desc_signature = desc[:100] if len(desc) > 100 else desc
        if desc and (desc_signature not in details):
            doc_parts.append(f"## Program Description\n{desc}")
            
        if details:
            doc_parts.append(f"## Scholarship Details\n{details}")
            
        if eligibility:
            doc_parts.append(f"## Eligibility Criteria\n{eligibility}")
            
        return "\n\n".join(doc_parts)

    def build_metadata(self, row):
        funding_cat = self._safe_meta(row.get('funding_category'))
        
        raw_amount = row.get('funding_amount')
        if pd.isna(raw_amount) or str(raw_amount).strip().lower() in ['nan', 'none', 'n/a', '']:
            if funding_cat == 'Fully Funded':
                funding_amount = 'Full Coverage'
            elif funding_cat == 'Partially Funded':
                funding_amount = 'Partial Coverage'
            else:
                funding_amount = 'Variable'
        else:
            funding_amount = self._safe_meta(raw_amount)

        return {
            "scholarship_name": self._safe_meta(row.get('scholarship_name')),
            "host_country": self._safe_meta(row.get('host_country')),
            "eligible_nationality": self._safe_meta(row.get('eligible_nationality')),
            "academic_level": self._safe_meta(row.get('academic_level')),
            "academic_major": self._safe_meta(row.get('academic_major')),
            "funding_category": funding_cat,
            "funding_amount": funding_amount,
            "standardized_deadline": self._safe_meta(row.get('standardized_deadline')),
            "scholarship_status": self._safe_meta(row.get('scholarship_status'), "Unknown"),
            "application_link": self._safe_meta(row.get('application_link'))
        }