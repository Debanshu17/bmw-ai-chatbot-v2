# Resume Bullets: Production-Grade RAG Engineering

These bullets follow the **Google XYZ format** (*"Accomplished [X] as measured by [Y], by doing [Z]"*), derived strictly from real, reproducible evaluation benchmark data (`reports/eval_v2_production_20261023_152551.json` vs. `reports/eval_v1_baseline_20261023_151530.json`).

---

### Bullet 1: Accuracy & RAG Architecture
> **Architected an enterprise RAG system with strict chunk provenance, elevating query accuracy from 7.5% to 92.5% (+85.0% absolute gain) across a 40-question technical benchmark by pairing Google Gemini 3,072-dimensional vector embeddings with Pydantic structured output validation and in-code citation verification.**

*Key Metrics Backing This Bullet*:
- Benchmark Accuracy: 92.5% (v2) vs. 7.5% (v1 baseline).
- Citation Precision: 100.0% (zero hallucinated citations).
- Citation Recall: 98.8%.

---

### Bullet 2: Guardrails & Hallucination Elimination
> **Eliminated hallucinated responses on out-of-scope queries, achieving a 100.0% Fallback True-Positive Rate (up from 0.0%) and 0.0% False-Fallback Rate by designing a dual-stage deterministic code guardrail that short-circuits LLM execution when cosine similarity falls below an empirically calibrated threshold of 0.72.**

*Key Metrics Backing This Bullet*:
- Fallback True-Positive Rate (TPR): 100.0% (10/10 out-of-scope queries intercepted).
- False-Fallback Rate (FFR): 0.0% (0/30 in-domain queries falsely rejected).
- Intercepted query token cost: $0.000 (100% LLM cost reduction on unanswerable prompts).

---

### Bullet 3: Latency & Cost Optimization Telemetry
> **Optimized system latency and operational observability, slashing p95 response time by 68.9% (from 8,976 ms to 2,792 ms) and maintaining an average query cost of $0.000037 by engineering an asynchronous execution telemetry pipeline and a lightweight NumPy vector search engine.**

*Key Metrics Backing This Bullet*:
- p95 Latency: Reduced from 8,976.1 ms to 2,792.3 ms (-6,183.8 ms improvement).
- Average Latency: Reduced from 1,538.8 ms to 1,023.6 ms (-515.2 ms).
- Cost Efficiency: $0.000037 average cost per technical query ($0.001467 total for 40 questions).
- Real-time Telemetry: Live TTFT, token consumption, and dynamic pricing metrics via `/metrics` endpoint.
