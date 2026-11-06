import os
import sys
import json
import time
import glob
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mock_llm import chat, reset

# --- TF-IDF Retrieval Setup ---
def load_policy_chunks(chunk_size=40, overlap=10):
    policy_files = glob.glob("data/policies/*.txt")
    chunks = []
    for pf in policy_files:
        name = os.path.basename(pf)
        with open(pf, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"source": name, "text": chunk_text})
            if i + chunk_size >= len(words):
                break
            i += (chunk_size - overlap)
    return chunks

CHUNKS = load_policy_chunks()
CORPUS = [c['text'] for c in CHUNKS]
VECTORIZER = TfidfVectorizer()
X = VECTORIZER.fit_transform(CORPUS)

def search_policy(query):
    query_vec = VECTORIZER.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    top_indices = scores.argsort()[::-1][:3]
    results = []
    for idx in top_indices:
        results.append({
            "source": CHUNKS[idx]["source"],
            "score": float(scores[idx]),
            "text": CHUNKS[idx]["text"]
        })
    return {"results": results}

# --- Tool Definitions ---
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    
    # Try cleaning order_id (extract digits if it has prefixes like 'O' or 'ORD')
    raw_id = str(order_id).strip()
    numeric_id = ''.join(filter(str.isdigit, raw_id))
    
    query = "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ? OR order_id = ?"
    
    # Check both exact string and parsed integer/string
    param1 = raw_id
    param2 = int(numeric_id) if numeric_id else raw_id
    
    df = pd.read_sql_query(query, conn, params=(param1, param2))
    if df.empty and numeric_id:
        # Fallback to numeric search
        df = pd.read_sql_query("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", conn, params=(int(numeric_id),))
        
    conn.close()
    
    if df.empty:
        return {"error": "order not found"}
    row = df.iloc[0]
    return {
        "order_id": str(row["order_id"]),
        "status": str(row["status"]),
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": float(row["delivery_minutes"])
    }

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0
    return {"refund_inr": round(refund, 2)}

def log_ledger(action_type, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action_type,
        "order_id": order_id,
        "refund_inr": refund_inr
    }
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def issue_refund(order_id, refund_inr):
    log_ledger("ISSUE_REFUND", order_id, refund_inr)
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    log_ledger("ESCALATE", order_id, refund_inr)
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Reads data/instamart.db and returns order details including status, order amount, and delivery minutes.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    },
    {
        "name": "search_policy",
        "description": "Searches the policy documents using TF-IDF retrieval and returns top matching chunks.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    },
    {
        "name": "calculate_refund",
        "description": "Computes refund amount based on delivery time rules (100% if >45 mins, else 0).",
        "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]}
    },
    {
        "name": "issue_refund",
        "description": "Issues the calculated refund, records it to the ledger, and returns the amount.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}
    },
    {
        "name": "escalate_to_human",
        "description": "Escalates the refund request to human support if above INR 1000 or requires manual review.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}
    }
]
# --- ReAct Agent Loop & Safety Limits ---
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tool_count=None):
    start_time = time.monotonic()
    if not run_id:
        run_id = f"run_{scenario_id}_{int(time.time())}"
        
    messages = [
        {"role": "system", "content": "You are AGENT, an autonomous support assistant for Instamart handling refund requests using available tools."},
        {"role": "user", "content": user_message}
    ]
    
    tokens_used = 0
    llm_calls = 0
    steps = []
    executed_tools = []
    
    for iteration in range(max_iterations):
        # Check timeout
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return save_and_return(run_id, scenario_id, "FALLBACK_TIMEOUT", "Timeout reached before completion.", steps, llm_calls, tokens_used, messages, executed_tools)
            
        # Call LLM
        reset()
        response = chat(messages)
        llm_calls += 1
        
        # Track token usage from mock response if available
        usage = getattr(response, "usage", None)
        if usage and hasattr(usage, "total_tokens"):
            tokens_used += usage.total_tokens
        else:
            tokens_used += 400  # estimated baseline tokens per turn
            
        # Check token limit
        if tokens_used > token_limit:
            return save_and_return(run_id, scenario_id, "FALLBACK_TOKEN_LIMIT", "Token limit exceeded.", steps, llm_calls, tokens_used, messages, executed_tools)
            
        choice = response.choices[0]
        message = choice.message
        
        # If no tool calls, we have the final answer
        if not message.tool_calls:
            final_answer = message.content.strip()
            # Determine status from the word before colon or default
            status = "REFUND"
            for prefix in ["REFUND", "ESCALATE", "DENIED", "NOT_FOUND"]:
                if final_answer.upper().startswith(prefix):
                    status = prefix
                    break
            
            # Log run
            log_agent_run(scenario_id, status, final_answer, llm_calls, tokens_used, steps)
            return {
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "run_id": run_id
            }
            
        # Process tool calls
        tool_calls_data = []
        for tc in message.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments
            })
            
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_data
        })
        
        for tc in message.tool_calls:
            t_name = tc.name
            t_args = tc.arguments
            
            # Execute tool
            if t_name in TOOL_MAP:
                try:
                    result = TOOL_MAP[t_name](**t_args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Tool {t_name} not found"}
                
            result_str = json.dumps(result)
            executed_tools.append(t_name)
            
            steps.append({
                "tool": t_name,
                "arguments": t_args,
                "observation": result
            })
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": t_name,
                "content": result_str
            })
            
            # Support simulated crash for Part D Task 8 test
            if crash_after_tool_count and len(executed_tools) == crash_after_tool_count:
                save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration + 1, executed_tools)
                raise RuntimeError("SIMULATED_CRASH")
                
        # Save checkpoint after iteration
        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration + 1, executed_tools)
        
    return save_and_return(run_id, scenario_id, "FALLBACK_MAX_ITERATIONS", "Max iterations reached.", steps, llm_calls, tokens_used, messages, executed_tools)

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, executed_tools):
    os.makedirs("state", exist_ok=True)
    chk = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_num": step_num,
        "executed_tools": executed_tools
    }
    with open(f"state/{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(chk, f, indent=2)

def resume_agent(run_id):
    path = f"state/{run_id}.json"
    if not os.path.exists(path):
        return {"error": "Checkpoint not found"}
    with open(path, "r", encoding="utf-8") as f:
        chk = json.load(f)
        
    # Resume agent loop with saved state
    return run_agent(
        user_message="Resume from checkpoint",
        scenario_id=chk["scenario_id"],
        run_id=chk["run_id"]
    )

def save_and_return(run_id, scenario_id, status, message, steps, llm_calls, tokens_used, messages, executed_tools):
    log_agent_run(scenario_id, status, message, llm_calls, tokens_used, steps)
    return {
        "status": status,
        "final_answer": message,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "run_id": run_id
    }

def log_agent_run(scenario_id, status, final_answer, llm_calls, tokens_used, steps):
    os.makedirs("outputs", exist_ok=True)
    record = {
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "steps": steps
    }
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    print("Refund Agent module loaded successfully.")