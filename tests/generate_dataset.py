import os
import json
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv 

load_dotenv()

class TestCase(BaseModel):
    name: str = Field(description="Name of the test case")
    question: str = Field(description="User intent or prompt")
    scholarship_context: str = Field(description="Detailed text of the scholarship requirements.")
    student_profile: str = Field(description="JSON string representing the student profile")
    ground_truth: str = Field(description="The ideal requirements the generated document MUST meet.")

class Dataset(BaseModel):
    cases: List[TestCase]

def generate_synthetic_dataset():
    print("Initializing Data Generator (Groq Llama)...")
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.7, 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    categories = [
        {
            "type": "High Match",
            "instruction": "Generate EXACTLY 10 distinct test cases where the student's profile perfectly matches the scholarship requirements. The scholarship should be technical (AI, Data, CS). The student must have high GPA (3.5+)."
        },
        {
            "type": "Partial Match",
            "instruction": "Generate EXACTLY 10 distinct test cases where the student's profile partially matches. For example, a Software Engineering student applying for a Business/Management scholarship. The ground truth should emphasize bridging the gap."
        },
        {
            "type": "Edge Cases (Extreme)",
            "instruction": "Generate EXACTLY 10 distinct extreme edge cases. Examples: A student with missing data, a student with a 1.2 GPA applying to Harvard, or a blatant contradiction."
        }
    ]

    all_test_cases = []

    for idx, category in enumerate(categories):
        print(f"Generating Batch {idx + 1}: {category['type']}...")
        
        # تم مضاعفة الأقواس المتعرجة هنا لمنع حدوث خطأ الـ Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI data engineer. Your task is to generate highly realistic synthetic data for testing a Scholarship RAG system. You MUST output a valid JSON object matching this exact structure: {{\"cases\": [{{\"name\": \"...\", \"question\": \"...\", \"scholarship_context\": \"...\", \"student_profile\": \"...\", \"ground_truth\": \"...\"}}]}} with EXACTLY 10 cases. Do not include any extra keys outside 'cases'."),
            ("user", "{instruction}")
        ])
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({"instruction": category["instruction"]})
            content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            raw_data = json.loads(content)
            validated_dataset = Dataset(**raw_data)
            
            for case in validated_dataset.cases:
                formatted_case = {
                    "name": case.name,
                    "question": case.question,
                    "contexts": [case.scholarship_context],
                    "ground_truth": case.ground_truth,
                    "student_profile": case.student_profile
                }
                all_test_cases.append(formatted_case)
            print(f"Successfully generated batch {idx + 1} ({len(validated_dataset.cases)} cases).")
        except Exception as e:
            print(f"Error generating {category['type']}: {e}")

    output_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_test_cases, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccessfully generated total {len(all_test_cases)} test cases!")
    print(f"Dataset saved to: {output_path}")

if __name__ == "__main__":
    generate_synthetic_dataset()