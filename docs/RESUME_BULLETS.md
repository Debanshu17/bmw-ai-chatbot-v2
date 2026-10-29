# Resume Bullets: Production-Grade RAG Engineering

These bullets follow the **Google XYZ format** (*"Accomplished [X] as measured by [Y], by doing [Z]"*), derived strictly from real, reproducible evaluation benchmark data (`reports/eval_v2_production_20261029_105018.json` vs. `reports/eval_v1_baseline_20261024_011731.json`).

---

### Bullet 1: Accuracy & RAG Architecture
> **Architected an enterprise RAG system with strict chunk provenance, elevating query accuracy from 0.0% to 90.0% (+90.0% absolute gain) across a 40-question technical benchmark by pairing Google Gemini 3,072-dimensional vector embeddings with Pydantic structured output validation and in-code citation verification (achieving 100.0% citation precision and 98.8% recall).**

*Key Metrics Backing This Bullet*:
- Benchmark Accuracy: **90.0%** (v2 production) vs. **0.0%** (No-RAG baseline).
- Citation Precision: **100.0%** (zero hallucinated or ungrounded citations).
- Citation Recall: **98.8%** across all in-scope technical queries.

---

### Bullet 2: Guardrails & Hallucination Elimination
> **Eliminated hallucinated responses on out-of-scope queries, achieving a 100.0% Fallback True-Positive Rate (up from 0.0%) and a 0.0% False-Fallback Rate by designing a dual-stage deterministic code guardrail that short-circuits LLM execution when cosine similarity falls below an empirically calibrated threshold of 0.72 (intercepting queries at $0.000 cost).**

*Key Metrics Backing This Bullet*:
- Fallback True-Positive Rate (TPR): **100.0%** (10/10 out-of-scope queries intercepted).
- False-Fallback Rate (FFR): **0.0%** (0/30 in-domain queries falsely rejected).
- Intercepted query token cost: **$0.000** (100% LLM cost reduction on unanswerable prompts).

---

### Bullet 3: Latency & Cost Optimization Telemetry
> **Optimized system latency and operational observability, delivering sub-second tail response times (p95 of 971.0 ms, average 799.2 ms) and an ultra-low query cost of $0.000013 by engineering an asynchronous execution telemetry pipeline and a lightweight NumPy vector search engine.**

*Key Metrics Backing This Bullet*:
- p95 Latency: **971.0 ms** across 40 technical evaluation queries.
- Average Latency: **799.2 ms** average query turnaround.
- Cost Efficiency: **$0.000013** average cost per technical query at paid-tier rates (actual development spend was **$0.00** on Google Free Tier).
- Real-time Telemetry: Live TTFT, token consumption, and dynamic pricing metrics via `/metrics` endpoint.
