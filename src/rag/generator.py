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
        
       
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.1,
            # temperature=0.2,        
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
        and formats the final text with ChromaDB metadata using parameterized queries.
        """
        if not docs:
            return ""

        scholarship_names = [doc.metadata.get('scholarship_name') for doc in docs]
        valid_names = [name for name in scholarship_names if name]
        
        application_processes = {}
        if valid_names:
            conn = duckdb.connect(self.duckdb_path)
            try:
                placeholders = ", ".join(["?"] * len(valid_names))
                
                query = f"""
                    SELECT scholarship_name, application_process 
                    FROM scholarships 
                    WHERE scholarship_name IN ({placeholders})
                """
                results = conn.execute(query, valid_names).fetchall()
                application_processes = {row[0]: row[1] for row in results}
            finally:
                conn.close()

        formatted_blocks = []
        for doc in docs:
            meta = doc.metadata
            name = meta.get('scholarship_name', 'Unknown Scholarship')
            
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
#     def _enrich_and_format_context(self, docs: List[Document]) -> str:
#         """
#         Late Enrichment: Fetches the application_process from DuckDB 
#         and formats the final text with ChromaDB metadata.
#         """
#         if not docs:
#             return ""

#         # Extract scholarship names to fetch their application process from DuckDB
#         scholarship_names = [doc.metadata.get('scholarship_name') for doc in docs]
        
#         # Safely handle single quote issues in SQL
#         safe_names = [name.replace("'", "''") for name in scholarship_names if name]
#         names_tuple = tuple(safe_names)
        
#         application_processes = {}
#         if names_tuple:
#             conn = duckdb.connect(self.duckdb_path)
#             try:
#                 # Handle tuple syntax for single element (e.g., ('Name',) -> ('Name'))
#                 in_clause = str(names_tuple) if len(names_tuple) > 1 else f"('{names_tuple[0]}')"
                
#                 query = f"""
#                     SELECT scholarship_name, application_process 
#                     FROM scholarships 
#                     WHERE scholarship_name IN {in_clause}
#                 """
#                 results = conn.execute(query).fetchall()
#                 application_processes = {row[0]: row[1] for row in results}
#             finally:
#                 conn.close()

#         # Format the rich context string
#         formatted_blocks = []
#         for doc in docs:
#             meta = doc.metadata
#             name = meta.get('scholarship_name', 'Unknown Scholarship')
            
#             # Construct a highly structured block for the LLM
#             block = f"""
# ### Scholarship: {name}
# - Host Country: {meta.get('host_country', 'Not specified')}
# - Funding Category: {meta.get('funding_category', 'Not specified')}
# - Amount: {meta.get('funding_amount', 'Not specified')}
# - Deadline: {meta.get('standardized_deadline', 'Not specified')}
# - Link: {meta.get('application_link', 'No link provided')}

# Description:
# {doc.page_content}

# Application Process:
# {application_processes.get(name, 'No specific application steps provided.')}
# """
#             formatted_blocks.append(block)

#         return "\n---\n".join(formatted_blocks)

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

    # cv analysing
    def parse_cv_text(self, cv_text: str) -> dict:
        system_prompt = """
        You are an expert data extractor and academic evaluator.
        Analyze the provided CV text and extract specific academic details.
        
        CRITICAL INSTRUCTIONS:
        1. Output ONLY a valid, strictly formatted JSON object. 
        2. Do NOT include any preambles, explanations, or markdown blocks (e.g., do not use ```json).
        3. Do NOT invent data. If a specific detail is not found in the text, use the specified default values.
        
        REQUIRED JSON STRUCTURE & DEFAULTS:
        {
            "nationality": "String (Extract the country. If not found, output 'Unknown')",
            "academic_level": "String (e.g., 'Bachelor', 'Master', 'PhD'. If not found, output 'Unknown')",
            "academic_major": "String (Extract the specific academic major. If not found, output 'Unknown')",
            "gpa": Float (Extract the GPA and STRICTLY CONVERT IT to a standard 4.0 scale. If not found, output 0.0),
            "research_interests": "String (A concise summary of technical skills or research focus. If not found, output null)",
            "target_countries": ["String"] (Extract preferred countries for study if mentioned. If not found, output []),
            "skills": ["String"] (Extract technical tools, programming languages, or soft skills. If not found, output []),
            "age": Integer (Extract the applicant's age if explicitly mentioned. If not found, output null)
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CV TEXT TO ANALYZE:\n{cv_text}"}
        ]
        
        try:
            response = self.llm.invoke(messages)
            raw_json_str = response.content.replace("```json", "").replace("```", "").strip()
            import json
            profile_data = json.loads(raw_json_str)
            return profile_data
            
        except Exception as e:
            print(f"Error in CV parsing: {e}")
            return {
                "nationality": "Unknown",
                "academic_level": "Bachelor",
                "academic_major": "Unknown",
                "gpa": 0.0,
                "research_interests": "",
                "target_countries": [],
                "skills": [],
                "age": None
            }

    #  generate motivation letters
    def generate_sop(self, profile_data: dict, scholarship_details: str, additional_notes: str = "", tone: str = "Professional", document_type: str = "Statement of Purpose") -> str:
        """
        Generates a highly tailored document by combining the student's profile 
        with the specific scholarship requirements, matching the requested tone and type.
        """
        system_prompt = f"""
        You are an elite academic admissions consultant and expert copywriter.
        Your task is to write a highly compelling {document_type} for a scholarship application.
        The overall tone of the text MUST be: {tone}.

        CRITICAL WRITING RULES:
        1. NO CLICHES: Never start with "My name is..." or "I am writing to apply for...". Start with a strong hook.
        2. SHOW, DON'T TELL: Demonstrate how the student's background directly aligns with the scholarship's goals.
        3. TAILORING: The letter MUST explicitly reference details from the "Scholarship Details". Explain WHY this specific program is the perfect fit.
        4. FORMATTING BY TYPE: Strictly adapt the structure to the requested document type. If it is an "Email", you MUST include a clear Subject Line. If it is a formal "Letter", use appropriate academic formatting.
        5. OUTPUT: Return ONLY the final letter text formatted in clean Markdown. Do not include any introductory remarks.
        
        CRITICAL GROUNDING RULES (ZERO HALLUCINATION):
        - STRICT TRUTH: Use ONLY the provided student profile and scholarship details.
        - NO EXTENSION: Do NOT fabricate past experiences, projects, awards, or personal backstory not explicitly mentioned in the profile.
        - ADAPTIVITY: If a piece of information (like volunteer work) is missing, do not invent it. Focus on articulating the student's motivation using ONLY the available facts.
        """

        user_content = f"""
        STUDENT PROFILE:
        {profile_data}

        SCHOLARSHIP DETAILS:
        {scholarship_details}

        ADDITIONAL NOTES (Key points to include):
        {additional_notes}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self.llm.invoke(messages)
            output_text = response.content.replace("```markdown", "").replace("```", "").strip()
            return output_text
        except Exception as e:
            print(f"Error generating document: {e}")
            raise ValueError(f"Failed to generate the {document_type}.")



    def chat_orchestrator(self, profile_data: dict, chat_history: list, retrieved_docs: list, user_message: str, intent: str = None) -> str:
        formatted_context = self._enrich_and_format_context(retrieved_docs)
        
        history_text = ""
        for msg in chat_history:
            history_text += f"{msg.role.capitalize()}: {msg.content}\n"

        system_prompt = """
        You are an elite Academic Scholarship Advisor. Assist the student clearly and concisely, utilizing the provided context and chat history.

        CRITICAL RULES:
        1. CONTEXT & MEMORY: Refer to the 'Chat History' for continuity. Base all factual claims, deadlines, and links STRICTLY on the 'Retrieved Scholarships Context'.
        2. ZERO HALLUCINATION: If a deadline, link, or specific requirement is missing in the context, explicitly state that it is not provided. Do not guess.
        3. SMART FILTERING (CRUCIAL): When explaining application steps or eligibility, you MUST filter the information based on the 'STUDENT PROFILE'. If the student is a 'Master' level, strictly ignore any requirements mentioned in the context meant for 'PhD' or 'Postdoc' applicants (e.g., dissertations, postdoctoral invitations).
        4. INTENT HANDLING:
           - 'compare': Generate a clean Markdown table comparing funding, deadlines, and levels of the mentioned scholarships.
           - 'roadmap': Provide a step-by-step application timeline based ONLY on the 'Application Process' in the context, tailored to the student's level.
           - 'explain': Bullet-point the specific eligibility criteria and required documents.
        5. CONCISENESS: Keep general responses brief and impactful to avoid overwhelming the student. Use clean Markdown formatting.
        6. LANGUAGE: Respond in the exact language used by the user in the 'CURRENT MESSAGE'.
        """

        user_content = f"""
        STUDENT PROFILE:
        {profile_data}
        
        CHAT HISTORY:
        {history_text}
        
        RETRIEVED SCHOLARSHIPS CONTEXT:
        {formatted_context}
        
        CURRENT MESSAGE:
        {user_message}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logging.error(f"Error in chat orchestrator: {e}")
            raise ValueError("Failed to generate chat response.")