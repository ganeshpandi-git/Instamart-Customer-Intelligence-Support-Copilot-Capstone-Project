import os
import sys
import glob
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mock_llm import chat, reset

def run_rag_evaluation():
    os.makedirs("outputs/part_d", exist_ok=True)
    
    # Load policy files
    policy_files = glob.glob("data/policies/*.txt")
    policies = {}
    for pf in policy_files:
        name = os.path.basename(pf)
        with open(pf, "r", encoding="utf-8") as f:
            policies[name] = f.read()
            
    print(f"Loaded {len(policies)} policy documents.")
    
    # Load evaluation questions
    eval_path = "data/raw/policy_eval.csv" if os.path.exists("data/raw/policy_eval.csv") else "data/policy_eval.csv"
    eval_df = pd.read_csv(eval_path)
    print(f"Loaded {len(eval_df)} evaluation questions. Columns found: {eval_df.columns.tolist()}")
    
    # Dynamically find the file/policy column
    possible_cols = ['policy_file', 'file', 'filename', 'document', 'doc']
    target_col = next((col for col in possible_cols if col in eval_df.columns), None)
    
    if not target_col:
        fallback_cols = [c for c in eval_df.columns if c not in ['id', 'question', 'expected_fact']]
        target_col = fallback_cols[0] if fallback_cols else eval_df.columns[2]
    
    print(f"Using column '{target_col}' for policy document lookup.")
    
    reset()
    results = []
    
    for idx, row in eval_df.iterrows():
        qid = row.get('id', idx)
        question = row['question']
        target_file = str(row[target_col]).strip()
        expected = row['expected_fact']
        
        context = policies.get(target_file, "")
        
        prompt = (
            f"You are a helpful policy assistant for Instamart. Answer the user's question using ONLY the provided policy context.\n\n"
            f"POLICY CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            f"Answer concisely and directly."
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = chat(messages)
        answer = response.choices[0].message.content.strip()
        
        # Safely handle NaN/float in expected_fact
        expected_str = str(expected).lower() if pd.notna(expected) else ""
        success = expected_str in answer.lower() if expected_str else False
        
        results.append({
            'id': qid,
            'question': question,
            'expected_fact': expected,
            'answer': answer,
            'passed': success
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/part_d/rag_eval_results.csv", index=False)
    accuracy = res_df['passed'].mean() * 100
    print(f"RAG Evaluation completed. Accuracy: {accuracy:.1f}%. Saved to outputs/part_d/rag_eval_results.csv")

if __name__ == "__main__":
    run_rag_evaluation()