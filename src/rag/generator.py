"""
Generation Layer for AI Scholarship System.

Implements the "Late Enrichment" pattern by fetching application processes 
from DuckDB, formatting the metadata, and strictly prompting Llama-3 
to assist the student without hallucinating.
"""

import os
import sys
import logging
import duckdb
from dotenv import load_dotenv
from typing import List, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables securely
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.schemas.student import StudentProfile

class ScholarshipGenerator:
    def __init__(self):
        logging.info("Initializing Generation Layer (Groq Llama-3)...")
        self.duckdb_path = os.path.join(PROJECT_ROOT, "db", "analytical.duckdb")
        
        # Ensure GROQ_API_KEY is in your environment variables
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", # Fast and powerful
            temperature=0.2,        # Low temperature to prevent hallucination
            max_tokens=1500
        )
        
# The Dynamic, Interactive, and Professional System Prompt
        self.prompt_template = PromptTemplate(
            input_variables=["student_profile", "context", "user_query"],
            template="""You are an elite, highly interactive Academic Scholarship Advisor. Your tone is professional, encouraging, and directly tailored to the student's specific needs.
Your primary goal is to answer the student's query based STRICTLY on the provided 'Context' scholarships.

---
Student Profile:
{student_profile}

Student's Query:
{user_query}

Retrieved Scholarships Context:
{context}
---

CRITICAL INSTRUCTIONS & RULES:

1. DYNAMIC & INTENT-DRIVEN RESPONSE: Tailor your response EXACTLY to what the user is asking:
   - If they ask for GENERAL RECOMMENDATIONS: Briefly list the matching scholarships, explain precisely *why* they fit their profile/major, provide the funding amount and deadline, and include the official link. (DO NOT list the application steps unless explicitly asked).
   - If they ask HOW TO APPLY: Focus heavily on explaining the "Application Process" steps from the context for the specific scholarship they mentioned.
   - If they ask about ELIGIBILITY/CONDITIONS: Focus exclusively on the specific requirements, demographics, and constraints of the scholarships.
   - Make it a natural conversation. Ask them at the end if they want to know more about the application steps for a specific scholarship.

2. ZERO HALLUCINATION: Base your answer EXCLUSIVELY on the 'Context'. Never invent names, deadlines, links, or application steps. If the context lacks the answer, clearly state: "لا تتوفر لدي تفاصيل حول هذه النقطة، يُرجى مراجعة الرابط الرسمي".

3. ELIGIBILITY DOUBLE-CHECK: Critically analyze the description. If a scholarship specifies demographic requirements (e.g., specific race, gender, minority status) that DO NOT clearly match the 'Student Profile', you MUST add a clear WARNING or exclude it.

4. STRUCTURED YET NATURAL OUTPUT: Use clean Markdown (bullet points, bold text) to make your response highly readable, but keep the text flowing naturally like a human advisor.

5. TONE & LANGUAGE: You MUST respond in the EXACT SAME LANGUAGE as the 'Student's Query'. If the query is in English, reply in English. If it is in French, reply in French. Do not let the student's nationality change the language of your response.

Response:"""
        )

    def _enrich_and_format_context(self, docs: List[Document]) -> str:
        """
        Late Enrichment: Fetches the application_process from DuckDB 
        and formats the final text with ChromaDB metadata.
        """
        if not docs:
            return ""

        # Extract scholarship names to fetch their application process from DuckDB
        scholarship_names = [doc.metadata.get('scholarship_name') for doc in docs]
        
        # Safely handle single quote issues in SQL
        safe_names = [name.replace("'", "''") for name in scholarship_names if name]
        names_tuple = tuple(safe_names)
        
        application_processes = {}
        if names_tuple:
            conn = duckdb.connect(self.duckdb_path)
            try:
                # Handle tuple syntax for single element (e.g., ('Name',) -> ('Name'))
                in_clause = str(names_tuple) if len(names_tuple) > 1 else f"('{names_tuple[0]}')"
                
                query = f"""
                    SELECT scholarship_name, application_process 
                    FROM scholarships 
                    WHERE scholarship_name IN {in_clause}
                """
                results = conn.execute(query).fetchall()
                application_processes = {row[0]: row[1] for row in results}
            finally:
                conn.close()

        # Format the rich context string
        formatted_blocks = []
        for doc in docs:
            meta = doc.metadata
            name = meta.get('scholarship_name', 'Unknown Scholarship')
            
            # Construct a highly structured block for the LLM
            block = f"""
### Scholarship: {name}
- Host Country: {meta.get('host_country', 'Not specified')}
- Funding Category: {meta.get('funding_category', 'Not specified')}
- Amount: {meta.get('funding_amount', 'Not specified')}
- Deadline: {meta.get('standardized_deadline', 'Not specified')}
- Link: {meta.get('application_link', 'No link provided')}

Description:
{doc.page_content}

Application Process:
{application_processes.get(name, 'No specific application steps provided.')}
"""
            formatted_blocks.append(block)

        return "\n---\n".join(formatted_blocks)

    def generate_response(self, profile: StudentProfile, retrieved_docs: List[Document], user_query: str) -> str:
        """
        Executes the LLM generation pipeline.
        """
        logging.info("Enriching context with DuckDB data and formatting...")
        formatted_context = self._enrich_and_format_context(retrieved_docs)
        
        # Format the student profile for the prompt
        profile_str = f"Major: {profile.academic_major}, Level: {profile.academic_level}, Nationality: {profile.nationality}"
        if profile.research_interests:
            profile_str += f"\nInterests: {profile.research_interests}"
            
        logging.info("Generating response via Groq LLM...")
        prompt_val = self.prompt_template.format(
            student_profile=profile_str,
            context=formatted_context,
            user_query=user_query
        )
        
        # Call the LLM
        response = self.llm.invoke(prompt_val)
        return response.content