import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import random
import pandas as pd
from mock_llm import chat, reset, est_tokens, TransientLLMError

def run_triage():
    os.makedirs("outputs", exist_ok=True)
    
    tickets_df = pd.read_csv("data/clean/tickets.csv", keep_default_na=False)
    segments_df = pd.read_csv("outputs/customer_segments.csv")
    premium_set = set(segments_df[segments_df['is_premium'] == True]['customer_id'])
    
    # Regular expressions for PII redaction
    email_pat = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
    phone_pat = re.compile(r'\b[6-9]\d{9}\b')
    order_pat = re.compile(r'ORD-\d{6}')
    
    email_count, phone_count, order_count = 0, 0, 0
    
    redacted_texts = []
    for txt in tickets_df['text']:
        if not txt.strip():
            redacted_texts.append("")
            continue
        e_matches = email_pat.findall(txt)
        p_matches = phone_pat.findall(txt)
        o_matches = order_pat.findall(txt)
        
        email_count += len(e_matches)
        phone_count += len(p_matches)
        order_count += len(o_matches)
        
        t1 = email_pat.sub("[EMAIL]", txt)
        t2 = phone_pat.sub("[PHONE]", t1)
        t3 = order_pat.sub("[ORDER_ID]", t2)
        redacted_texts.append(t3)
        
    tickets_df['redacted_text'] = redacted_texts
    total_pii = email_count + phone_count + order_count
    print(f"PII Redacted -> Emails: {email_count}, Phones: {phone_count}, Order IDs: {order_count}, Total: {total_pii}")
    
    # Few-shot system prompt
    few_shot_prompt = (
        "CLASSIFY. You are a support ticket classifier. Allowed categories: billing, delivery, technical, fraud. "
        "You must reply ONLY with a JSON object of the form {\"category\": \"...\", \"reason\": \"...\"}.\n"
        "EXAMPLES:\n"
        "Text: 'Someone hacked my account and used my wallet balance. This is fraud.' -> {\"category\": \"fraud\", \"reason\": \"keyword match\"}\n"
        "Text: 'I was charged twice for order ORD-123456.' -> {\"category\": \"billing\", \"reason\": \"keyword match\"}"
    )
    
    reset()
    results = []
    
    for idx, row in tickets_df.iterrows():
        tid = row['ticket_id']
        cid = row['customer_id']
        txt = row['redacted_text']
        true_cat = row['true_category']
        
        # Input guardrails
        if not txt.strip():
            results.append({
                'ticket_id': tid, 'customer_id': cid, 'status': 'skipped_empty',
                'category': '', 'segment_name': '', 'priority': 'NORMAL', 'route': 'human_review', 'tokens_used': 0
            })
            continue
            
        if est_tokens(txt) > 400:
            results.append({
                'ticket_id': tid, 'customer_id': cid, 'status': 'flagged_long',
                'category': '', 'segment_name': '', 'priority': 'NORMAL', 'route': 'human_review', 'tokens_used': 0
            })
            continue
            
        # Model call with retries
        messages = [{"role": "system", "content": few_shot_prompt}, {"role": "user", "content": txt}]
        response = None
        success = False
        
        for attempt in range(3):
            try:
                response = chat(messages)
                success = True
                break
            except TransientLLMError as e:
                wait = 0.05 * (2 ** attempt) * random.uniform(0.9, 1.15)
                print(f"Retry {attempt} for {tid} due to error: {e}, waiting {wait:.4f}s")
                
        if not success:
            results.append({
                'ticket_id': tid, 'customer_id': cid, 'status': 'flagged_invalid_output',
                'category': '', 'segment_name': '', 'priority': 'NORMAL', 'route': 'human_review', 'tokens_used': 0
            })
            continue
            
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        # Validate output
        cat = None
        status = 'flagged_invalid_output'
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("category") in ["billing", "delivery", "technical", "fraud"]:
                cat = parsed["category"]
                status = 'classified_json'
        except json.JSONDecodeError:
            words = re.findall(r"[a-z]+", content.lower())
            allowed = ["billing", "delivery", "technical", "fraud"]
            matches = [w for w in allowed if w in words]
            if len(matches) == 1:
                cat = matches[0]
                status = 'classified_keyword'
                
        # Routing logic
        seg_row = segments_df[segments_df['customer_id'] == cid]
        seg_name = seg_row['segment_name'].values[0] if len(seg_row) > 0 else 'Unknown'
        is_prem = cid in premium_set
        
        if status in ['skipped_empty', 'flagged_long', 'flagged_invalid_output']:
            route, priority = 'human_review', 'NORMAL'
        elif cat == 'fraud':
            route, priority = 'security_team', 'HIGH'
        elif is_prem:
            route, priority = 'premium_desk', 'HIGH'
        else:
            route, priority = 'support_queue', 'NORMAL'
            
        results.append({
            'ticket_id': tid, 'customer_id': cid, 'status': status,
            'category': cat if cat else '', 'segment_name': seg_name,
            'priority': priority, 'route': route, 'tokens_used': tokens
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/triage_results.csv", index=False)
    print(f"Triage completed. Results saved to outputs/triage_results.csv. Total rows: {len(res_df)}")

if __name__ == "__main__":
    run_triage()