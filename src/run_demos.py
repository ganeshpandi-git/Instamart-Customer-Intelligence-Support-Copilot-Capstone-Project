import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from refund_agent import run_agent, resume_agent

def run_demonstrations():
    print("--- Running Safety & Resumability Demos ---")
    
    # 1. Token Limit Demo
    print("\n[Demo 1] S1 with token_limit=200:")
    res_token = run_agent("Please refund order ORD-110093 because it arrived late.", scenario_id="S1_TOKEN_DEMO", token_limit=200)
    print(f"Result Status: {res_token['status']}")
    
    # 2. Timeout Demo
    print("\n[Demo 2] S1 with time_limit_seconds=0:")
    res_time = run_agent("Please refund order ORD-110093 because it arrived late.", scenario_id="S1_TIME_DEMO", time_limit_seconds=0)
    print(f"Result Status: {res_time['status']}")
    
    # 3. Crash and Resume Demo
    print("\n[Demo 3] S1 Crash and Resume:")
    run_id = "run_S1_crash_demo"
    tools_before_crash = []
    try:
        run_agent(
            user_message="Please refund order ORD-110093 because it arrived late.",
            scenario_id="S1",
            run_id=run_id,
            crash_after_tool_count=2
        )
    except RuntimeError as e:
        print(f"Caught simulated crash: {e}")
        
    # Resume the agent
    print(f"Resuming agent with run_id: {run_id}")
    res_resume = resume_agent(run_id)
    print(f"Resume Final Status: {res_resume['status']}")
    print(f"Resume Final Answer: {res_resume['final_answer']}")
    
    # Collect tools executed across both parts
    if os.path.exists(f"state/{run_id}.json"):
        import json
        with open(f"state/{run_id}.json", "r") as f:
            chk = json.load(f)
            print(f"Tools executed in order: {chk['executed_tools']}")

if __name__ == "__main__":
    run_demonstrations()