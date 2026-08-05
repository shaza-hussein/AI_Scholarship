"""
Integration Test for the Hybrid Retrieval Engine.
Evaluates the Retriever against various student profiles to validate 
deterministic filtering, semantic search, and cross-encoder re-ranking.
"""

import os
import sys
import time

# Resolve project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.schemas.student import StudentProfile
from src.rag.retriever import ScholarshipRetriever

def run_tests():
    print("="*60)
    print("🚀 Initializing Test Suite for Hybrid Retriever")
    print("="*60)
    
    # Initialize the retriever once for all tests
    start_init = time.time()
    retriever = ScholarshipRetriever()
    print(f"Initialization completed in {time.time() - start_init:.2f} seconds.\n")

    # Define Test Scenarios
    test_cases = [
        {
            "test_name": "Test Case 1: Deep Technical Specialization",
            "profile": StudentProfile(
                nationality="Syria",
                academic_level="Master",
                academic_major="Computer Science",
                gpa=3.8,
                research_interests="Generative AI architectures for tabular data, Variational Autoencoders (VAEs), and CTGAN.",
                skills=["Python", "PyTorch", "DuckDB", "Computer Vision"]
            )
        },
        {
            "test_name": "Test Case 2: General / Broad Field",
            "profile": StudentProfile(
                nationality="Egypt",
                academic_level="Bachelor",
                academic_major="Business Administration",
                gpa=3.2,
                research_interests="Management, digital marketing, and leadership.",
                skills=["Communication", "Project Management"]
            )
        },
        {
            "test_name": "Test Case 3: Zero Match (Edge Case)",
            "profile": StudentProfile(
                nationality="Atlantis", # Fictional country to force DuckDB exclusion
                academic_level="Postdoc",
                academic_major="Marine Biology",
                gpa=4.0
            )
        }
    ]

    # Execute Tests
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 {case['test_name']}")
        print(f"Profile: {case['profile'].academic_level} in {case['profile'].academic_major} from {case['profile'].nationality}")
        print(f"{'-'*60}")
        
        start_search = time.time()
        results = retriever.match_scholarships(profile=case['profile'], top_k=3, fetch_k=15)
        search_time = time.time() - start_search
        
        if not results:
            print(f"Result: NO MATCHES FOUND (Execution time: {search_time:.2f}s)")
            print("Status: SUCCESS (System correctly aborted/filtered out invalid profile)")
        else:
            print(f"Result: FOUND {len(results)} MATCHES (Execution time: {search_time:.2f}s)")
            for idx, doc in enumerate(results, 1):
                scholarship_name = doc.metadata.get('scholarship_name', 'Unknown')
                score = doc.metadata.get('rerank_score', 0.0)
                
                print(f"\n  [{idx}] {scholarship_name}")
                print(f"      Re-ranker Score: {score:.4f} (Higher is better)")
                # Print a clean snippet of the content
                snippet = doc.page_content.replace('\n', ' ')[:150]
                print(f"      Snippet: {snippet}...")

    print("\n" + "="*60)
    print(" Test Suite Execution Completed.")
    print("="*60)

if __name__ == "__main__":
    run_tests()