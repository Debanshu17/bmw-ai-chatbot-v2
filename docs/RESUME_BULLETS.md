# Resume Bullets: Production-Grade RAG Engineering

These bullets adhere strictly to the **Google XYZ format** (*"Accomplished [X] as measured by [Y], by doing [Z]"*), derived exclusively from the final validated benchmark reports ([`eval_v1_baseline_20261031_010339.json`](../reports/eval_v1_baseline_20261031_010339.json), [`eval_v2_production_20261031_133931.json`](../reports/eval_v2_production_20261031_133931.json), and dedicated latency reports [`eval_v1_latency_bench_20261031.json`](../reports/eval_v1_latency_bench_20261031.json), [`eval_v2_latency_bench_20261031.json`](../reports/eval_v2_latency_bench_20261031.json)). No figures are simulated or extrapolated.

---

### Option 1: Accuracy, Retrieval Precision & Hallucination Elimination
> **Architected an enterprise RAG system for a 32-chunk technical knowledge base across 3 BMW M engineering guides, elevating factual accuracy from 45.0% to 100.0% (and 100.0% vs. 53.3% on answerable queries) against a closed-book Gemini 3.5 Flash Lite baseline by implementing 3072-dimensional vector retrieval, in-code provenance verification (100% citation precision, 98.75% recall), and deterministic fallback gating.**

### Option 2: Deterministic Guardrails & Out-of-Scope Fallback Handling
> **Eliminated model hallucinations on unanswerable and out-of-scope queries, achieving 100.0% refusal compliance (vs. 20.0% for baseline) and a 0.0% false-fallback rate by engineering dual-stage guardrails (pre-LLM similarity threshold gating at 0.72 and post-generation citation validation) that short-circuited 100% of adversarial questions at \$0 token cost.**

### Option 3: Production Telemetry & Resilient Evaluation Infrastructure
> **Engineered an automated evaluation harness (v2.1.0) and operational telemetry pipeline that identified and resolved a production citation-key mismatch bug, reducing false fallbacks from 3.33% to 0.0%; built client-side rate-limiting and stop-and-wait exponential backoff that completed a 40-question benchmark with 0 infrastructure errors and 100% genuine LLM generation (0 extractive fallback).**

---

### Key Measured Metrics Reference

| Metric | Closed-Book Baseline | v2 Production RAG | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (All 40 Qs)** | 45.0% (18/40) | **100.0%** (40/40) | **+55.0%** |
| **Answerable Questions Accuracy (30 Qs)** | 53.3% (16/30) | **100.0%** (30/30) | **+46.7%** |
| **Out-of-Scope Questions Accuracy (10 Qs)** | 20.0% (2/10) | **100.0%** (10/10) | **+80.0%** |
| **Citation Precision** | 100.0% | **100.0%** | 0.0% |
| **Citation Recall** | 25.0% | **98.75%** | **+73.75%** |
| **Fallback True-Positive Rate (TPR)** | 0.0% (0/10) | **100.0%** (10/10) | **+100.0%** |
| **False-Fallback Rate (FFR)** | 0.0% (0/30) | **0.0%** (0/30) | **0.0%** |
| **Answerable Median (p50) Compute Latency** | 3,999.2 ms | **8,190.8 ms** | +4,191.6 ms |
| **Answerable Average Compute Latency** | 5,453.9 ms | **7,355.5 ms** | +1,901.6 ms |
| **Answerable p95 Compute Latency** | 13,606.1 ms | **15,208.2 ms** | +1,602.1 ms |
| **Average Cost / Query (Overall)** | \$0.000668 | **\$0.000627** | -\$0.000041 |
| **— Average Cost / Query (Answerable Only)** | \$0.000668 | **\$0.000836** | +\$0.000168 |
| **Infrastructure Errors** | 0 (0.0%) | **0 (0.0%)** | Zero errors |
| **Extractive Fallback Used** | 0 / 40 | **0 / 40 (0%)** | 100% genuine LLM generation |

*Latency & Overhead Note: Across answerable questions, v2 is computationally heavier and slower than the unaugmented baseline due to 3072-dimensional query embedding generation, vector similarity retrieval, context expansion (~850 vs ~25 prompt tokens), and in-code citation provenance validation. In exchange, v2 achieves 100.0% factual accuracy (vs. 53.3% baseline), 98.75% citation recall (vs. 25.0% baseline), and zero hallucinations.*
