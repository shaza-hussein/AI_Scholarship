import re
import pandas as pd

class RagDocumentBuilder:
    def __init__(self):
        self.html_pattern = re.compile(r'<[^>]+>')
        self.separator_pattern = re.compile(r'-{3,}|={3,}|\*{3,}')
        self.bullet_pattern = re.compile(r'[•-]\s*\n')
        self.newline_pattern = re.compile(r'\n{3,}')
        
    def clean_text(self, text, min_words=10):
        if pd.isna(text) or str(text).strip().lower() in ['nan', 'none', 'n/a', '']:
            return ""
            
        text = str(text)
        text = self.html_pattern.sub(' ', text)
        text = self.separator_pattern.sub(' ', text)
        text = self.bullet_pattern.sub('- ', text)
        text = self.newline_pattern.sub('\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        cleaned_text = text.strip()
        
        if len(cleaned_text.split()) < min_words:
            return ""
            
        return cleaned_text

    def build_document(self, row):
        doc_parts = []
        
        desc = self.clean_text(row.get('description', ''))
        details = self.clean_text(row.get('scholarship_details', ''))
        eligibility = self.clean_text(row.get('eligibility', ''))
        
        if desc and (desc not in details):
            doc_parts.append(f"## Program Description\n{desc}")
            
        if details:
            doc_parts.append(f"## Scholarship Details\n{details}")
            
        if eligibility:
            doc_parts.append(f"## Eligibility Criteria\n{eligibility}")
            
        return "\n\n".join(doc_parts)

    def build_metadata(self, row):
        return {
            "scholarship_name": str(row.get('scholarship_name', '')),
            "host_country": str(row.get('host_country', 'Unspecified')),
            "eligible_nationality": str(row.get('eligible_nationality', 'Unspecified')),
            "academic_level": str(row.get('academic_level', 'Unspecified')),
            "academic_major": str(row.get('academic_major', 'Unspecified')),
            "funding_category": str(row.get('funding_category', 'Unspecified')),
            "funding_amount": str(row.get('funding_amount', 'Unspecified')),
            "standardized_deadline": str(row.get('standardized_deadline', 'Not Specified')),
            "scholarship_status": str(row.get('scholarship_status', 'Unknown')),
            "application_link": str(row.get('application_link', ''))
        }