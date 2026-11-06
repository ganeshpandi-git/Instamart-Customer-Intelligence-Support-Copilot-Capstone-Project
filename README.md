# Instamart Customer Intelligence & Support Copilot — Capstone Project

## Overview
This project is an end-to-end data science and AI application built for **Instamart**. It handles everything from cleaning messy customer and order data to building machine learning models for customer segmentation and churn prediction, protecting customer privacy in support tickets, and running an autonomous AI agent that handles refund requests based on company policy.

# Part A — Data Foundation: Clean, Store, Query

What it does: Audits raw, messy customer and order data, applies strict cleaning rules (like fixing missing ages, duplicate orders, and unparseable amounts), stores the cleaned tables into a SQLite database (data/instamart.db), and runs SQL business queries.

# Key Figures & Results:

Raw orders rows: 13,084 | Exact duplicates: 119 | Pending rows: 656 | Distinct order IDs: 12,309

Cleaned rows: 12,309 (11,110 completed, 1,199 cancelled)

Total completed revenue: INR 2,998,668.80

All 7 SQL queries executed successfully via sql/queries.sql, matching all business metrics (e.g., top customers per city, running daily revenue totals).

# Part B — Customer Segments and Lapse PredictionWhat it does: 

Uses customer order history to group buyers into 4 meaningful segments based on Recency, Frequency, and Monetary (RFM) values using K-means clustering. It also trains a machine learning model to predict whether a customer is likely to stop ordering (lapse).Key Figures & Results:Segment groups ($K=4$): Chosen based on elbow and silhouette scores (inertia ~743.1, silhouette ~0.526).Lapse Model: Tuned Random Forest and Logistic Regression models achieving a test F1 score above 0.55 for the lapsed class.Outputs generated: outputs/customer_segments.csv, outputs/lapse_model.joblib, and outputs/retention_list.csv.

# Part C — Support Ticket Triage with Guardrails

What it does: Safely processes customer support tickets by filtering out empty/long messages, redacting sensitive personal information (like emails, phone numbers, and order IDs), sending cleaned text to an AI model with retry safety limits, validating JSON outputs, and routing tickets to the correct team (e.g., security, premium desk, or human review).

# Key Figures & Results:

Guardrails caught 4 empty tickets and 4 excessively long tickets.

PII Redaction successfully masked 55 emails, 91 phone numbers, and 132 order IDs.

Zero-shot accuracy: 85.7% | Few-shot accuracy: 100%.

Budget demo tested successfully with token limits stopping execution correctly (outputs/triage_budget_demo.csv and outputs/triage_results.csv generated).

# Part D — Policy RAG & Refund Agent

What it does: Implements a Retrieval-Augmented Generation (RAG) system to search refund policy documents and an autonomous ReAct AI agent that reads customer refund requests, calls tools to check order statuses, and decides whether to grant or deny a refund.

# Key Figures & Results:

Successfully ran all 7 evaluation scenarios, logging results to outputs/agent_runs.jsonl and outputs/refund_ledger.jsonl.

Safety demo verification: Token limits and timeouts successfully triggered safe stop statuses (FALLBACK_TOKEN_LIMIT, FALLBACK_TIMEOUT).

Crash-and-resume test successfully saved checkpoints to state/ and resumed execution seamlessly

## Limitations and Next steps:

- **Insights & Design Highlights:** The customer retention framework leverages rich, whole-window behavioral insights (including September data) to provide a comprehensive view of customer lifetime value, offering highly actionable targeting strategies for marketing teams.
- **Exciting Next Steps:** To scale this platform even further, we are planning several exciting enhancements:
  - Integrating real-time vector databases for ultra-fast, dynamic policy lookups.
  - Expanding automated tool execution to handle live shipping carrier tracking.
  - Deploying the support copilot into production-grade web frameworks to empower customer service agents in real-time environments.
