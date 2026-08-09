import os
import json
import uuid
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def run_retrieval_evaluation():
    print("Initializing Retrieval Evaluation...")
    
    # 1. Load the synthetic dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 2. Initialize local embeddings
    print("Loading Embeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 3. Prepare Documents (30 Real + 70 Noise)
    print("Injecting 100 documents into ChromaDB...")
    docs = []
    
    # Inject real docs
    for idx, case in enumerate(test_cases):
        doc_id = f"real_doc_{idx}"
        case["target_doc_id"] = doc_id  # Save target ID for evaluation
        
        # We store the scholarship context as the document content
        doc = Document(
            page_content=case["contexts"][0] if isinstance(case.get("contexts"), list) else case.get("scholarship_context", ""),
            metadata={"doc_id": doc_id, "type": "real", "case_name": case["name"]}
        )
        docs.append(doc)

    # Inject 70 noise docs (e.g., cooking, sports, irrelevant grants)
    noise_topics = ["Italian cooking recipes", "Football match results", "History of the Roman Empire", "Basic car maintenance", "Quantum computing for dummies"]
    for i in range(70):
        noise_content = f"This is a random document about {noise_topics[i % len(noise_topics)]}. It contains noise data to distract the retriever. Unique ID: {uuid.uuid4()}"
        doc = Document(
            page_content=noise_content,
            metadata={"doc_id": f"noise_doc_{i}", "type": "noise"}
        )
        docs.append(doc)

    # 4. Create Ephemeral Chroma VectorStore
    vector_store = Chroma.from_documents(docs, embeddings)
    print("ChromaDB Indexing Complete.\n")

    # 5. Execute Evaluation
    print("Running Queries and Calculating Metrics (K=5)...\n")
    results = []
    
    for case in test_cases:
        query = case["question"]
        target_id = case["target_doc_id"]
        
        # Retrieve Top 5
        retrieved_docs = vector_store.similarity_search(query, k=5)
        retrieved_ids = [d.metadata["doc_id"] for d in retrieved_docs]
        
        # Calculate Metrics
        # Recall@5: 1.0 if the target doc is in the top 5, else 0.0
        recall_at_5 = 1.0 if target_id in retrieved_ids else 0.0
        
        # MRR (Mean Reciprocal Rank): 1 / rank position (1st = 1.0, 2nd = 0.5, etc.)
        mrr = 0.0
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            mrr = 1.0 / rank
            
        results.append({
            "Test Case": case["name"],
            "Recall@5": recall_at_5,
            "MRR": mrr,
            "Retrieved IDs": retrieved_ids
        })

    # 6. Display Results
    df = pd.DataFrame(results)
    
    print("=" * 60)
    print("RETRIEVAL EVALUATION DASHBOARD (ChromaDB)")
    print("=" * 60)
    
    for index, row in df.iterrows():
        print(f"CASE: {row['Test Case']}")
        print(f"Recall@5 : {row['Recall@5']:.2f}")
        print(f"MRR      : {row['MRR']:.2f}")
        print("-" * 60)
        
    print(f"\nAVERAGE RECALL@5 : {df['Recall@5'].mean():.2f}")
    print(f"AVERAGE MRR      : {df['MRR'].mean():.2f}")

if __name__ == "__main__":
    run_retrieval_evaluation()