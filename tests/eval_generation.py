import sys
from unittest.mock import MagicMock

# MONKEY PATCH: Inject dummy module to bypass Ragas legacy import bug
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from src.api.dependencies import init_ai_engines, get_generator

load_dotenv()

def run_generation_evaluation():
    print("Initializing Generation Evaluation (Faithfulness Only)...")
    
    # 1. Load the synthetic dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 2. Init AI Engines for generation
    init_ai_engines()
    generator = get_generator()
    
    # 3. Setup Ragas Judge (Llama-3.1)
    print("Connecting to Groq (Llama-3.1-8b-instant) as the Judge...")
    evaluator_llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=3
    )
    
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    data = {
        "question": [],
        "answer": [],
        "contexts": []
    }

    # 4. Generate SOPs from hardcoded perfect context
    print(f"Generating documents for {len(test_cases)} test cases. Please wait...")
    for idx, case in enumerate(test_cases):
        
        # Read profile data based on your JSON structure
        # Safely parse profile data with fallback for LLM edge-case hallucinations
        raw_profile = case["student_profile"]
        if isinstance(raw_profile, dict):
            profile_data = raw_profile
        else:
            try:
                # Attempt to parse valid JSON
                profile_data = json.loads(raw_profile)
            except json.JSONDecodeError:
                # Fallback: If Llama-3 generated plain text or empty string, wrap it in a dict
                profile_data = {"raw_extracted_text": raw_profile}
        
        scholarship_context = case["contexts"][0] if isinstance(case.get("contexts"), list) else case.get("scholarship_context", "")
        
        # Generate the response
        answer = generator.generate_sop(
            profile_data=profile_data,
            scholarship_details=scholarship_context,
            additional_notes=case["question"],
            tone="Professional",
            document_type="Statement of Purpose"
        )
        
        data["question"].append(case["question"])
        data["answer"].append(answer)
        
        # CRITICAL FIX 2.0: Dynamically unpack ALL profile data into readable text to prevent Information Asymmetry
        profile_facts = []
        if isinstance(profile_data, dict):
            for key, value in profile_data.items():
                clean_key = str(key).replace("_", " ").title()
                profile_facts.append(f"- {clean_key}: {value}")
        else:
            profile_facts.append(str(profile_data))
            
        profile_text = "Student Profile Information:\n" + "\n".join(profile_facts)
        combined_context = [scholarship_context, profile_text]
        
        data["contexts"].append(combined_context)

    dataset = Dataset.from_dict(data)

    # 5. Run Ragas Evaluation strictly for Faithfulness
    print("\nRunning Ragas Faithfulness Evaluation (Sequential)...")
    safe_config = RunConfig(max_workers=1, max_retries=3, timeout=60)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=safe_config
    )

    df = result.to_pandas()
    
    print("\n" + "=" * 60)
    print("GENERATION EVALUATION DASHBOARD (Faithfulness)")
    print("=" * 60)
    
    for index, row in df.iterrows():
        print(f"CASE: {test_cases[index]['name']}")
        print(f"Faithfulness Score : {row.get('faithfulness', 'N/A'):.2f} / 1.0")
        print("-" * 60)
        
    avg_faithfulness = df['faithfulness'].mean()
    print(f"\nAVERAGE FAITHFULNESS : {avg_faithfulness:.2f} / 1.0")

if __name__ == "__main__":
    run_generation_evaluation()