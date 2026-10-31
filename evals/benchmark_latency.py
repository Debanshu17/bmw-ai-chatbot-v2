"""Standalone Latency Benchmark Runner for Answerable Queries.

Measures pure pipeline compute latency (retrieval + inference) with client-side
rate-limiter pacing strictly executed OUTSIDE the timed measurement block.
Skips LLM-as-a-judge grading to conserve quota and measure exclusively generation latency.
"""

import json
import logging
from pathlib import Path
import sys
import time
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import settings
from evals.run_evals import startup_model_check, load_dataset, run_v1_baseline_query
from generation.generator import generate_rag_response
from retrieval.index import global_vector_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("latency_bench")


def run_latency_benchmark():
    dataset_file = BASE_DIR / "evals" / "dataset.jsonl"
    all_entries = load_dataset(dataset_file)
    answerable_entries = [e for e in all_entries if e.get("answerable", False)]

    logger.info(f"Loaded {len(answerable_entries)} answerable evaluation queries.")
    logger.info(f"Target model: {settings.CHAT_MODEL}")

    startup_model_check(settings.CHAT_MODEL, settings.GEMINI_API_KEY)
    global_vector_index.build_or_load()

    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Run v2 Production RAG Latency Benchmark (30 questions)
    # -------------------------------------------------------------
    logger.info("=== Starting v2 Production RAG Latency Measurement (Pacing Excluded) ===")
    v2_results = []
    for idx, entry in enumerate(answerable_entries, start=1):
        qid = entry["id"]
        q = entry["question"]
        logger.info(f"[v2 {idx}/30] Benchmarking {qid}...")

        resp = generate_rag_response(q)
        duration_ms = round(resp.telemetry.duration_s * 1000.0, 2)
        ttft_ms = round(resp.telemetry.ttft_ms, 2)

        v2_results.append({
            "id": qid,
            "question": q,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
            "is_error": resp.is_error,
            "prompt_tokens": resp.telemetry.prompt_tokens,
            "completion_tokens": resp.telemetry.completion_tokens,
            "answer_preview": resp.answer[:60]
        })
        logger.info(f"  -> Finished {qid} in {duration_ms} ms (TTFT: {ttft_ms} ms)")

    v2_durations = [r["duration_ms"] for r in v2_results if not r["is_error"]]
    v2_ttfts = [r["ttft_ms"] for r in v2_results if not r["is_error"]]

    v2_report = {
        "benchmark": "v2_production_answerable_latency",
        "model": settings.CHAT_MODEL,
        "sample_size": len(v2_durations),
        "rate_limiting_excluded_from_timer": True,
        "avg_latency_ms": round(float(np.mean(v2_durations)), 2),
        "p50_latency_ms": round(float(np.percentile(v2_durations, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(v2_durations, 95)), 2),
        "avg_ttft_ms": round(float(np.mean(v2_ttfts)), 2),
        "results": v2_results
    }

    v2_json_path = out_dir / "eval_v2_latency_bench_20261031.json"
    with open(v2_json_path, "w", encoding="utf-8") as f:
        json.dump(v2_report, f, indent=2)
    logger.info(f"v2 Latency benchmark saved to {v2_json_path}")

    # -------------------------------------------------------------
    # 2. Run Closed-Book Baseline Latency Benchmark (30 questions)
    # -------------------------------------------------------------
    logger.info("=== Starting Closed-Book Baseline Latency Measurement (Pacing Excluded) ===")
    v1_results = []
    for idx, entry in enumerate(answerable_entries, start=1):
        qid = entry["id"]
        q = entry["question"]
        logger.info(f"[Baseline {idx}/30] Benchmarking {qid}...")

        resp = run_v1_baseline_query(q)
        duration_ms = round(resp["telemetry"].duration_s * 1000.0, 2)
        ttft_ms = round(resp["telemetry"].ttft_ms, 2)

        v1_results.append({
            "id": qid,
            "question": q,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
            "is_error": resp["is_error"],
            "prompt_tokens": resp["prompt_tokens"],
            "completion_tokens": resp["completion_tokens"],
            "answer_preview": resp["answer"][:60]
        })
        logger.info(f"  -> Finished {qid} in {duration_ms} ms (TTFT: {ttft_ms} ms)")

    v1_durations = [r["duration_ms"] for r in v1_results if not r["is_error"]]
    v1_ttfts = [r["ttft_ms"] for r in v1_results if not r["is_error"]]

    v1_report = {
        "benchmark": "v1_baseline_answerable_latency",
        "model": settings.CHAT_MODEL,
        "sample_size": len(v1_durations),
        "rate_limiting_excluded_from_timer": True,
        "avg_latency_ms": round(float(np.mean(v1_durations)), 2),
        "p50_latency_ms": round(float(np.percentile(v1_durations, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(v1_durations, 95)), 2),
        "avg_ttft_ms": round(float(np.mean(v1_ttfts)), 2),
        "results": v1_results
    }

    v1_json_path = out_dir / "eval_v1_latency_bench_20261031.json"
    with open(v1_json_path, "w", encoding="utf-8") as f:
        json.dump(v1_report, f, indent=2)
    logger.info(f"Baseline Latency benchmark saved to {v1_json_path}")

    print("\n" + "=" * 60)
    print("PURE COMPUTE LATENCY BENCHMARK RESULTS (PACING EXCLUDED)")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Baseline (Answerable)':<20} | {'v2 RAG (Answerable)':<20}")
    print("-" * 70)
    print(f"{'Average Latency':<25} | {v1_report['avg_latency_ms']:>10.2f} ms        | {v2_report['avg_latency_ms']:>10.2f} ms")
    print(f"{'p50 Latency':<25} | {v1_report['p50_latency_ms']:>10.2f} ms        | {v2_report['p50_latency_ms']:>10.2f} ms")
    print(f"{'p95 Latency':<25} | {v1_report['p95_latency_ms']:>10.2f} ms        | {v2_report['p95_latency_ms']:>10.2f} ms")
    print(f"{'Average TTFT':<25} | {v1_report['avg_ttft_ms']:>10.2f} ms        | {v2_report['avg_ttft_ms']:>10.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    run_latency_benchmark()
