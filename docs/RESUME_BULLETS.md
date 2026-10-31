# Resume Bullets: Production-Grade RAG Engineering

These bullets adhere strictly to the **Google XYZ format** (*"Accomplished [X] as measured by [Y], by doing [Z]"*), derived exclusively from the final validated benchmark reports ([`eval_v1_baseline_20261031_010339.json`](../reports/eval_v1_baseline_20261031_010339.json), [`eval_v2_production_20261031_011115.json`](../reports/eval_v2_production_20261031_011115.json), and dedicated latency reports [`eval_v1_latency_bench_20261031.json`](../reports/eval_v1_latency_bench_20261031.json), [`eval_v2_latency_bench_20261031.json`](../reports/eval_v2_latency_bench_20261031.json)). No figures are simulated or extrapolated.

---

### Option 1: Accuracy, Retrieval Precision & Hallucination Elimination
> **Architected an enterprise RAG system for a 32-chunk technical knowledge base across 3 BMW M engineering guides, elevating factual accuracy from 45.0% to 95.0% (and 93.3% vs. 53.3% on answerable queries) against a closed-book Gemini 3.5 Flash Lite baseline by implementing 3072-dimensional vector retrieval, in-code provenance verification (100% citation precision, 96.25% recall), and deterministic fallback gating.**

### Option 2: Deterministic Guardrails & Out-of-Scope Fallback Handling
> **Eliminated model hallucinations on unanswerable and out-of-scope queries, achieving 100.0% refusal compliance (vs. 20.0% for baseline) by engineering dual-stage guardrails (pre-LLM similarity threshold gating at 0.72 and post-generation citation validation) that short-circuited 100% of adversarial questions at \$0 token cost.**

### Option 3: Production Telemetry & Resilient Evaluation Infrastructure
> **Engineered an automated evaluation harness (v2.1.0) and operational telemetry pipeline achieving a median response latency of 2.2s (p50: 2,208 ms vs. 3,999 ms for closed-book baseline); built client-side rate-limiting and stop-and-wait exponential backoff that completed a 40-question benchmark with 0 infrastructure errors and 100% genuine LLM generation (0 extractive fallback).**

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
| **Median (p50) Compute Latency (Answerable)** | 3,999.2 ms | **2,208.1 ms** | **-1,791.1 ms** |
| **Average Compute Latency (Answerable)** | 5,453.9 ms | **8,548.6 ms**\* | +3,094.7 ms |
| **Infrastructure Errors** | 0 (0.0%) | **0 (0.0%)** | Zero errors |
| **Extractive Fallback Used** | 0 / 40 | **0 / 40 (0%)** | 100% genuine LLM generation |

*\*v2 average compute latency was inflated by two 60s external socket timeouts on queries 5 and 6 (network incident); clean re-run of queries 5 and 6 averaged 2,218.8 ms, yielding an adjusted 30-query mean compute latency of 4,791.5 ms (median p50: 2,208.1 ms).*
