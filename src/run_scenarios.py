import os
import pandas as pd
from refund_agent import run_agent, resume_agent

def execute_all():
    os.makedirs("outputs", exist_ok=True)
    
    # Clear previous logs
    for fname in ["outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl"]:
        if os.path.exists(fname):
            os.remove(fname)
            
    scenarios_path = "data/raw/agent_scenarios.csv" if os.path.exists("data/raw/agent_scenarios.csv") else "data/agent_scenarios.csv"
    if not os.path.exists(scenarios_path):
        print(f"Scenarios file not found at {scenarios_path}")
        return
        
    df = pd.read_csv(scenarios_path)
    print(f"Loaded {len(df)} scenarios.")
    
    for _, row in df.iterrows():
        sid = row['scenario_id']
        msg = row['message']
        print(f"Running scenario {sid}...")
        res = run_agent(msg, sid)
        print(f"--> {sid} Status: {res['status']} | Final: {res['final_answer']}")

if __name__ == "__main__":
    execute_all()