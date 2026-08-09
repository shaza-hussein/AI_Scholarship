import sys
from unittest.mock import MagicMock

# MONKEY PATCH: Inject a dummy module to bypass Ragas legacy import bug
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.api.dependencies import init_ai_engines, get_generator

def run_comprehensive_evaluation():
    print("Initializing AI Engines...")
    init_ai_engines()
    generator = get_generator()
    
    print("Connecting to Groq (Llama-3.1-8b-instant) as the Supreme Judge...")
    evaluator_llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=3
    )
    
    print("Loading Local Embeddings for Math operations...")
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # قراءة ملف الـ JSON المُولّد تلقائياً من مجلد tests
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    print(f"Loading test cases from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset file at {dataset_path}. Please run generate_dataset.py first.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        TEST_CASES = json.load(f)

    print(f"Successfully loaded {len(TEST_CASES)} test cases for evaluation.")

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [] 
    }

    print("Generating responses from ScholarAI for evaluation...")
    for idx, test in enumerate(TEST_CASES):
        print(f"Processing Test {idx + 1}: {test.get('name', 'Unnamed Case')}")
        
        # معالجة آمنة لـ student_profile لتجنب أخطاء الـ JSON
        student_profile_raw = test.get("student_profile", "{}")
        if isinstance(student_profile_raw, dict):
            profile_data = student_profile_raw
            profile_str = json.dumps(student_profile_raw)
        elif isinstance(student_profile_raw, str):
            try:
                profile_data = json.loads(student_profile_raw)
                profile_str = student_profile_raw
            except json.JSONDecodeError:
                profile_data = {"profile_raw": student_profile_raw}
                profile_str = student_profile_raw
        else:
            profile_data = {}
            profile_str = "{}"
        
        # استخراج تفاصيل المنحة من contexts
        contexts_source = test.get("contexts", [""])
        scholarship_detail = contexts_source[0] if isinstance(contexts_source, list) and len(contexts_source) > 0 else str(contexts_source)

        try:
            answer = generator.generate_sop(
                profile_data=profile_data,
                scholarship_details=scholarship_detail,
                additional_notes=test.get("question", ""),
                tone="Professional",
                document_type="Statement of Purpose"
            )
        except Exception as e:
            print(f"Error generating SOP for test {idx + 1}: {e}")
            answer = "Error generating response."
        
        data["question"].append(test.get("question", ""))
        
        # دمج بيانات الطالب مع المنحة في قائمة السياق لمنع اتهام النظام بالهلوسة
        contexts_list = test["contexts"] if isinstance(test["contexts"], list) else [test["contexts"]]
        combined_context = contexts_list + [profile_str]
        data["contexts"].append(combined_context)
        
        data["answer"].append(answer)
        data["ground_truth"].append(test.get("ground_truth", ""))

    dataset = Dataset.from_dict(data)

    print("\nRunning Ragas Evaluation (Faithfulness, Relevance, Precision, Recall)...")
    print("Please wait, executing sequentially to respect Groq Rate Limits...")
    
    safe_config = RunConfig(max_workers=1, max_retries=3, timeout=60)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=safe_config
    )

    df = result.to_pandas()
    
    print("\n" + "="*70)
    print("RAGAS COMPREHENSIVE EVALUATION DASHBOARD (TERMINAL)")
    print("="*70)
    
    for index, row in df.iterrows():
        case_name = TEST_CASES[index].get('name', f'Case {index + 1}')
        print(f"\nTEST CASE {index + 1}: {case_name}")
        print(f"1. Faithfulness (Groundedness)    : {row.get('faithfulness', 'N/A'):.2f} / 1.0 (No Hallucinations)")
        print(f"2. Answer Relevance               : {row.get('answer_relevancy', 'N/A'):.2f} / 1.0 (Met user intent)")
        print(f"3. Context Precision (Inc. MRR)   : {row.get('context_precision', 'N/A'):.2f} / 1.0 (Useful docs ranked high)")
        print(f"4. Context Recall                 : {row.get('context_recall', 'N/A'):.2f} / 1.0 (Brought all needed info)")
        print("-" * 70)
        
if __name__ == "__main__":
    run_comprehensive_evaluation()