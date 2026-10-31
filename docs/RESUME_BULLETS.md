# Resume Bullets: Production-Grade RAG Engineering

These bullets adhere strictly to the **Google XYZ format** (*"Accomplished [X] as measured by [Y], by doing [Z]"*), derived exclusively from the final validated benchmark reports ([`eval_v1_baseline_20261031_010339.json`](../reports/eval_v1_baseline_20261031_010339.json) and [`eval_v2_production_20261031_011115.json`](../reports/eval_v2_production_20261031_011115.json)). No figures are simulated or extrapolated.

---

### Option 1: Accuracy, Retrieval Precision & Hallucination Elimination
> **Architected an enterprise RAG system for dense automotive engineering manuals, elevating factual accuracy from 45.0% to 95.0% (and 93.3% vs. 53.3% on answerable queries) against a closed-book Gemini 3.5 Flash Lite baseline by implementing 3072-dimensional vector retrieval, in-code provenance verification (100% citation precision, 96.25% recall), and deterministic fallback gating.**

### Option 2: Deterministic Guardrails & Out-of-Scope Fallback Handling
> **Eliminated model hallucinations on unanswerable and out-of-scope queries, achieving 100.0% refusal compliance (vs. 20.0% for baseline) by engineering dual-stage guardrails (pre-LLM similarity threshold gating at 0.72 and post-generation citation validation) that short-circuited 100% of adversarial questions at \$0 token cost.**

### Option 3: Production Telemetry & Resilient Evaluation Infrastructure
> **Engineered an automated continuous evaluation harness (v2.1.0) and operational telemetry pipeline tracking TTFT (530.9 ms avg) and latency percentiles; built client-side rate-limiting and stop-and-wait exponential backoff that completed a 40-question benchmark with 0 infrastructure errors and 100% genuine LLM generation (0 extractive fallback).**

---

### Key Measured Metrics Reference

| Metric | Closed-Book Baseline | v2 Production RAG | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (All 40 Qs)** | 45.0% (18/40) | **95.0%** (38/40) | **+50.0%** |
| **Answerable Questions Accuracy (30 Qs)** | 53.3% (16/30) | **93.3%** (28/30) | **+40.0%** |
| **Out-of-Scope Questions Accuracy (10 Qs)** | 20.0% (2/10) | **100.0%** (10/10) | **+80.0%** |
| **Citation Precision** | 100.0% | **100.0%** | 0.0% |
| **Citation Recall** | 25.0% | **96.25%** | **+71.25%** |
| **Fallback True-Positive Rate (TPR)** | 0.0% (0/10) | **100.0%** (10/10) | **+100.0%** |
| **False-Fallback Rate (FFR)** | 0.0% (0/30) | **3.33%** (1/30) | +3.33% |
| **Average Time-To-First-Token (TTFT)** | N/A *(non-streaming)* | **530.9 ms** | Sub-second |
| **Infrastructure Errors** | 0 (0.0%) | **0 (0.0%)** | Zero errors |
| **Extractive Fallback Used** | 0 / 40 | **0 / 40 (0%)** | 100% genuine LLM generation |
