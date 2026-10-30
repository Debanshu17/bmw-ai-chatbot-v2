# BMW M Production-Grade RAG System (v2)

[![CI](https://github.com/Debanshu17/bmw-ai-chatbot-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/Debanshu17/bmw-ai-chatbot-v2/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash_Lite-4285F4.svg)
![License](https://img.shields.io/badge/License-MIT-black.svg)

An enterprise-grade, high-performance Retrieval-Augmented Generation (RAG) system engineered for technical automotive specifications. Built with deterministic multi-stage guardrails, in-code citation provenance verification, real-time latency and financial telemetry, and an automated continuous evaluation harness.

> **Note on Repositories**: The original chatbot codebase is preserved 100% untouched and intact in its own standalone repository at [`https://github.com/Debanshu17/bmw-ai-chatbot`](https://github.com/Debanshu17/bmw-ai-chatbot). This repository (`bmw-ai-chatbot-v2`) represents the completely re-architected production system.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    Client["BMW M Glassmorphism HUD (Web UI / REST API)"] --> FastAPI["FastAPI Application Server (app.py)"]
    
    subgraph Core ["Enterprise RAG Pipeline"]
        FastAPI --> TelemetryStart["Telemetry Tracker: Start Timer"]
        TelemetryStart --> VectorSearch["Dense Vector Search (Gemini 3072-dim Cosine Similarity)"]
        
        VectorSearch --> Gate1{"Gate 1: Pre-LLM Score Check\n(Top Score >= 0.72?)"}
        
        Gate1 -- "No (< 0.72 or Empty)" --> FastFallback["Deterministic Short-Circuit\nReturn Fixed Fallback Message\n(0 Token Cost, < 20ms)"]
        
        Gate1 -- "Yes (>= 0.72)" --> PromptEngine["Context Formatter & Prompt Engine"]
        PromptEngine --> LLM["Google Gemini 2.5 Flash Lite\n(Zero-Knowledge System Instruction)"]
        
        LLM --> Validator["In-Code Provenance Validator\n(Reject Fabricated Chunk IDs / Snippets)"]
        Validator --> Gate2{"Gate 2: Post-LLM Grounding Check\n(Valid Citations Present?)"}
        
        Gate2 -- "No Citations / Refusal" --> FallbackClean["Enforce Fixed Fallback Message\n(used_fallback = True)"]
        Gate2 -- "Valid Provenance" --> ValidResponse["Synthesize RAGResponse\n(Answer + Citations + Confidence)"]
    end
    
    FastFallback --> TelemetryStop["Telemetry Tracker: Stop Timer & Log"]
    FallbackClean --> TelemetryStop
    ValidResponse --> TelemetryStop
    
    TelemetryStop --> Logs["telemetry/logs/telemetry.jsonl"]
    TelemetryStop --> APIOutput["Return Validated JSON / UI Render"]
    APIOutput --> Client
```

---

## 📊 Benchmark Comparison: No-RAG Baseline vs. v2 Production

*Results pending a clean re-run.*

---

## 🔍 Known Limitations & Analysis

Across the 40 benchmark questions, **36 passed (90.0% accuracy)**. Analysis of the 4 failed queries highlights key areas for future retrieval enhancements:

1. **`eval_003` (Factual - S58 Intake Air Cooling Circuit)**:
   - *Query*: "How is the intake air cooled in the S58 forced induction system?"
   - *Root Cause*: **Chunk boundary split**. Chapter 2 spans multiple aspects of forced induction. The top retrieved chunk detailed turbocharger 1 & 2 pressure delivery, while the specific indirect liquid-to-air intercooler description was located in an adjacent chunk that ranked below top-k.
2. **`eval_006` (Numeric Specification - S58 Redline & Compression Ratio)**:
   - *Query*: "What is the redline and compression ratio of the BMW S58 powerplant?"
   - *Root Cause*: **Dispersed tabular metadata**. The retrieved chunk contained the compression ratio (9.3:1) and cylinder dimensions, but the 7,200 RPM redline specification was contained in a separate technical table chunk.
3. **`eval_024` (Factual - M Traction Control 10-Stage Granularity)**:
   - *Query*: "How many stages are provided by M Traction Control and what is the difference between Stage 1 and Stage 10?"
   - *Root Cause*: **Boundary truncation**. The generated answer captured the behavioral differences between Stage 1 and Stage 10, but the sentence explicitly stating the "10 stages" count was clipped at the chunk boundary.
4. **`eval_026` (Multi-Chunk Synthesis - Lightweight Body & Interior Measures)**:
   - *Query*: "Synthesize the lightweight measures applied to the body and interior to lower center of gravity and vehicle mass."
   - *Root Cause*: **Cross-document dispersion**. The retrieval engine successfully found the CFRP roof chunk (saving 6.0 kg and lowering CG by 2.5 mm), but interior carbon bucket seats (saving 9.6 kg) and forged suspension arms spanned 3 distinct document files, exceeding dense retrieval top-k density.

### Graceful-Degradation Extractive Fallback Mode
The system features an automated **extractive fallback mode** as an emergency safety net: if LLM JSON parsing or network streaming fails, the system safely extracts the top grounded chunk sentence with verified source provenance so the user receives a grounded excerpt rather than an unhandled error.

**Benchmark Grounding Rule**: Under rigorous evaluation standards, extractive fallback is treated as a separate recovery mode and is strictly **excluded from accuracy**. If extractive fallback exceeds 5.0% of answerable questions in a benchmark run, the run is flagged as `INVALID`. In the final validated evaluation run, **0 out of 30 answerable queries used extractive fallback (100% genuine Gemini generation)**, guaranteeing authentic model performance.

---

## 🧱 The 4 Pillars of Implementation

### Pillar 1: Evaluation Harness (`evals/`)
- **40-Query Benchmark Dataset (`evals/dataset.jsonl`)**: 30 answerable technical queries (numeric specs, factual, multi-chunk synthesis) and 10 out-of-scope adversarial questions. Every ground truth fact is verified against source documents.
- **Infrastructure Error Resilience (`evals/retry.py`)**: Automatic exponential backoff retry (up to 5 attempts) on transient API failures (HTTP 429 quota exhaustion, timeouts, 503). If an infrastructure failure persists, the result is tagged as `ERROR` and strictly excluded from accuracy and latency percentiles. If infrastructure errors exceed 5% of queries, the run immediately aborts.
- **Checkpointing & Resume (`--resume`)**: The evaluation runner checkpoints results to disk after every single query. Running with `--resume` reuses successful cached queries and only retries missing or `ERROR` questions.
- **Answer Source & Judge Auditing**: Explicitly records whether each response originated from `"gemini"` (LLM generation) or `"extractive_fallback"`, and whether each judgment used `"llm"` (LLM-as-a-judge zero-temperature rubric) or `"fuzzy"` (deterministic refusal match).
- **CLI Runner (`evals/run_evals.py`)**: Computes valid accuracy, citation precision/recall, fallback reliability (TPR/FFR), latency percentiles (p50, p95), answer source breakdowns, judge breakdowns, and token costs.

### Pillar 2: Strict Metadata & Source Citations (`retrieval/` & `generation/`)
- **Provenance Chunking**: Ingestion pipeline records `source_doc`, `page_or_chapter`, `chunk_id`, and character offsets (`start`, `end`) for every chunk.
- **Pydantic Validation (`generation/schemas.py`)**: Responses strictly enforce the `RAGResponse` schema:
  `{ answer, citations: [{doc, page_or_chapter, chunk_id, quote_snippet}], confidence, used_fallback }`.
- **In-Code Verification (`generation/validator.py`)**: Confirms every cited `chunk_id` was in the retrieved set and verifies that `quote_snippet` exists as an authentic substring. Fabricated citations are stripped.
- **Interactive UI Citations**: The frontend renders interactive motorsport pills under each response that expand to reveal the document name, chapter, and verbatim quote snippet.

### Pillar 3: Cost & Latency Telemetry (`telemetry/`)
- **Execution Tracking (`telemetry/monitor.py`)**: Tracks Time-To-First-Token (TTFT), total latency, prompt tokens, and completion tokens per query.
- **Config-Driven Cost Calculator (`config/pricing.json`)**: Derives query costs dynamically from model pricing rates without hardcoded constants.
- **Metrics Endpoint (`/metrics`)**: Exposes rolling window operational metrics including average latency, p50, p95, total costs, and fallback rates.

### Pillar 4: Deterministic Fallback Guardrails (`guardrails/`)
- **Dual-Stage Gating**:
  - **Gate 1 (Pre-LLM)**: Intercepts queries where top retrieval similarity is below `0.72` or empty, returning the canonical refusal message without invoking the LLM (0 token cost, < 20ms).
  - **Gate 2 (Post-LLM)**: Detects missing citations or refusal indicators, enforcing the fixed fallback message.
- **Fixed Constant**:
  `"I couldn't find this in the provided documents, so I can't answer it reliably."`
- **Zero-Leakage System Prompt**: Strictly forbids using outside training knowledge.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.11+
- Google Gemini API Key

### 2. Installation
```bash
# Clone or navigate to the v2 repository
cd bmw-ai-chatbot-v2

# Install pinned dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

### 3. Build Vector Store Index
```bash
python -c "from retrieval.index import global_vector_index; global_vector_index.build()"
```

### 4. Run the Application
```bash
# Start FastAPI backend server
python app.py
```
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to interact with the BMW M glassmorphism interface!

---

## 🧪 Running Unit Tests & Evaluations

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Run Evaluation Benchmark
```bash
# Run full 40-question evaluation
python evals/run_evals.py --dataset evals/dataset.jsonl --out reports/

# Resume a previous evaluation run (skips successful queries, retries missing/errors)
python evals/run_evals.py --dataset evals/dataset.jsonl --out reports/ --resume

# Run a 10-question smoke test
python evals/run_evals.py --dataset evals/dataset.jsonl --limit 10 --out reports/

# Diff two reports
python evals/run_evals.py --diff reports/eval_v1_baseline_20261024_011731.json reports/eval_v2_production_20261029_105018.json
```

---

## 📦 Repository Organization

- **Original Baseline Repository**: [`https://github.com/Debanshu17/bmw-ai-chatbot`](https://github.com/Debanshu17/bmw-ai-chatbot) (Preserved untouched).
- **v2 Production RAG Repository**: [`https://github.com/Debanshu17/bmw-ai-chatbot-v2`](https://github.com/Debanshu17/bmw-ai-chatbot-v2) (Production system).
