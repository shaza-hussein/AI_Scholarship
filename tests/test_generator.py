"""
Comprehensive End-to-End Test for the AI Scholarship System.
Tests multiple conversational intents (Discovery, Application, Eligibility) in English.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.schemas.student import StudentProfile
from src.rag.retriever import ScholarshipRetriever
from src.rag.generator import ScholarshipGenerator

def run_comprehensive_test():
    print("=" * 80)
    print("Initializing Comprehensive End-to-End System Test")
    print("=" * 80)
    
    start_init = time.time()
    retriever = ScholarshipRetriever()
    generator = ScholarshipGenerator()
    print(f"\n System Initialized in {time.time() - start_init:.2f} seconds.\n")

    # Mock Profile (Egyptian student, Bachelor in Business)
    profile = StudentProfile(
        nationality="Egypt",
        academic_level="Bachelor",
        academic_major="Business Administration",
        gpa=3.5,
        research_interests="Digital Marketing and Leadership"
    )
    
    print(" Step 1: Retrieving context for the student profile...")
    # Retrieve documents only once to save processing time
    docs = retriever.match_scholarships(profile, top_k=2)
    
    if not docs:
        print(" No matching scholarships found. Cannot proceed with generation tests.")
        return

    print(f" Found {len(docs)} matching scholarships. Proceeding to Generation Layer...\n")

    # Define the test queries targeting different intents (English)
    test_queries = [
        {
            "intent": "Discovery & Recommendations",
            "query": "I am an Egyptian student looking for a Bachelor's scholarship in Business Administration. Can you suggest suitable options based on my profile?"
        },
        {
            "intent": "Application Process Focus",
            "query": "What are the exact application steps and deadlines for the marketing or leadership scholarships you found?"
        },
        {
            "intent": "Eligibility & Constraints Focus",
            "query": "Are there any specific demographic requirements, racial constraints, or specific eligibility conditions I should be aware of for these scholarships?"
        }
    ]

    # Step 2: Generation Testing Loop
    for i, test in enumerate(test_queries, 1):
        print("=" * 80)
        print(f" Test Case {i}: {test['intent']}")
        print(f" User Query: {test['query']}")
        print("=" * 80)
        
        start_gen = time.time()
        response = generator.generate_response(profile, docs, test['query'])
        
        print("\n AI ASSISTANT RESPONSE ✨")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print(f"Generation Time: {time.time() - start_gen:.2f} seconds.\n")

if __name__ == "__main__":
    run_comprehensive_test()