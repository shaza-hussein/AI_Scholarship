import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.schemas.student import StudentProfile
from src.rag.retriever import ScholarshipRetriever
from src.rag.generator import ScholarshipGenerator

class MockMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

def run_chat_orchestrator_test():
    print("=" * 80)
    print("Initializing Comprehensive Chat Orchestrator Test")
    print("=" * 80)
    
    start_init = time.time()
    retriever = ScholarshipRetriever()
    generator = ScholarshipGenerator()
    print(f"\nSystem Initialized in {time.time() - start_init:.2f} seconds.\n")

    profile_dict = {
        "nationality": "Egypt",
        "academic_level": "Master",
        "academic_major": "Computer Science",
        "gpa": 3.8,
        "research_interests": "Artificial Intelligence and Machine Learning"
    }
    
    profile_obj = StudentProfile(**profile_dict)
    
    print("Step 1: Retrieving context for the student profile...")
    docs = retriever.match_scholarships(profile_obj, query="Artificial Intelligence", top_k=3)
    
    if not docs:
        print("No matching scholarships found. Cannot proceed with generation tests.")
        return

    print(f"Found {len(docs)} matching scholarships. Proceeding to Generation Layer...\n")

    test_turns = [
        {
            "intent": "None",
            "query": "Hello, I am looking for fully funded AI scholarships in Europe. What do you recommend?"
        },
        {
            "intent": "compare",
            "query": "Can you compare the scholarships you just mentioned in a table format?"
        },
        {
            "intent": "roadmap",
            "query": "I want to apply for the DAAD scholarship. Can you give me a step-by-step roadmap for the application process?"
        },
        {
            "intent": "None",
            "query": "Wait, what was the deadline for the first scholarship you mentioned? I forgot."
        }
    ]

    chat_history = []

    print("Step 2: Multi-turn Chat Simulation")
    for i, test in enumerate(test_turns, 1):
        print("=" * 80)
        print(f"Turn {i} | Intent: {test['intent']}")
        print(f"User: {test['query']}")
        print("=" * 80)
        
        start_gen = time.time()
        
        response = generator.chat_orchestrator(
            profile_data=profile_dict,
            chat_history=chat_history,
            retrieved_docs=docs,
            user_message=test['query'],
            intent=test['intent']
        )
        
        print("\nAI ASSISTANT RESPONSE")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print(f"Generation Time: {time.time() - start_gen:.2f} seconds.\n")
        
        chat_history.append(MockMessage(role="user", content=test['query']))
        chat_history.append(MockMessage(role="assistant", content=response))

if __name__ == "__main__":
    run_chat_orchestrator_test()