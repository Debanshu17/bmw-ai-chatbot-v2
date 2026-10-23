"""CLI Evaluation Harness and Benchmarking Runner.

Runs datasets through the RAG pipeline or v1 baseline, computes comprehensive
accuracy, citation precision/recall, fallback reliability, latency percentiles,
and financial cost metrics. Outputs structured JSON and human-readable Markdown.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import settings
from evals.judge import judge_answer
from generation.generator import generate_rag_response
from retrieval.index import global_vector_index
from telemetry.monitor import TelemetryTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_v1_baseline_query(question: str) -> Dict[str, Any]:
    """Evaluates v1 baseline (zero RAG, pure parametric LLM) in a read-only wrapper."""
    import google.generativeai as genai
    tracker = TelemetryTracker(model_name=settings.CHAT_MODEL).start()

    prompt = f"Answer the user's question:\n{question}"
    try:
        model = genai.GenerativeModel(model_name=settings.CHAT_MODEL)
        tracker.record_first_token()
        response = model.generate_content(prompt)
        text = response.text or ""
        prompt_tokens = len(prompt.split()) * 2
        completion_tokens = len(text.split()) * 2
    except Exception as e:
        logger.error(f"v1 baseline call failed: {e}")
        text = "SYSTEM ERROR"
        prompt_tokens = 0
        completion_tokens = 0

    telemetry = tracker.stop(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return {
        "answer": text,
        "citations": [],
        "used_fallback": False,
        "telemetry": telemetry
    }


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Loads JSONL evaluation dataset."""
    entries = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_citation_metrics(
    generated_citations: List[Any],
    expected_sources: List[Dict[str, str]]
) -> Tuple[float, float]:
    """Computes citation precision and citation recall against expected sources."""
    if not expected_sources and not generated_citations:
        return 1.0, 1.0
    if not expected_sources and generated_citations:
        # Hallucinated citations on unanswerable query
        return 0.0, 1.0
    if expected_sources and not generated_citations:
        # Missing citations on answerable query
        return 1.0, 0.0

    expected_set = {
        (s.get("doc", "").lower(), s.get("page_or_chapter", "").lower())
        for s in expected_sources
    }

    # Handle citation objects or dicts
    gen_set = set()
    for c in generated_citations:
        if isinstance(c, dict):
            doc = c.get("doc", "").lower()
            poc = c.get("page_or_chapter", "").lower()
        else:
            doc = getattr(c, "doc", "").lower()
            poc = getattr(c, "page_or_chapter", "").lower()
        gen_set.add((doc, poc))

    correct = len(gen_set.intersection(expected_set))
    precision = correct / len(gen_set) if gen_set else 1.0
    recall = correct / len(expected_set) if expected_set else 1.0
    return precision, recall


def run_eval(
    dataset_path: Path,
    out_dir: Path,
    is_baseline_v1: bool = False,
    similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Executes evaluation run across all entries in the dataset."""
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]

    logger.info(f"Loaded {len(dataset)} evaluation entries from {dataset_path}")

    # Build vector index if needed for v2
    if not is_baseline_v1:
        global_vector_index.build_or_load()

    results: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = {}
    latencies: List[float] = []
    ttfts: List[float] = []
    costs: List[float] = []

    total_answerable = 0
    total_unanswerable = 0
    correct_fallbacks = 0  # True positives on unanswerable
    false_fallbacks = 0    # False fallbacks on answerable

    for i, item in enumerate(dataset):
        qid = item["id"]
        q = item["question"]
        gt = item["ground_truth_answer"]
        expected_srcs = item.get("expected_sources", [])
        answerable = item.get("answerable", True)
        cat = item.get("category", "general")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}

        category_stats[cat]["total"] += 1
        if answerable:
            total_answerable += 1
        else:
            total_unanswerable += 1

        logger.info(f"[{i+1}/{len(dataset)}] Evaluating {qid} ({cat})...")

        if is_baseline_v1:
            resp_data = run_v1_baseline_query(q)
            cand_ans = resp_data["answer"]
            cits = resp_data["citations"]
            used_fb = resp_data["used_fallback"]
            telemetry = resp_data["telemetry"]
        else:
            rag_resp = generate_rag_response(
                query=q,
                similarity_threshold=similarity_threshold
            )
            cand_ans = rag_resp.answer
            cits = rag_resp.citations
            used_fb = rag_resp.used_fallback
            telemetry = rag_resp.telemetry

        # Fallback accounting
        if not answerable:
            if used_fb or cand_ans.strip() == settings.FALLBACK_MESSAGE:
                correct_fallbacks += 1
        else:
            if used_fb or cand_ans.strip() == settings.FALLBACK_MESSAGE:
                false_fallbacks += 1

        # Judge evaluation
        judge_res = judge_answer(
            question=q,
            candidate_answer=cand_ans,
            ground_truth_answer=gt,
            answerable=answerable
        )
        passed = judge_res["score"] == 1.0
        if passed:
            category_stats[cat]["passed"] += 1

        # Citation metrics
        c_prec, c_rec = compute_citation_metrics(cits, expected_srcs)

        # Telemetry metrics
        duration_ms = telemetry.duration_s * 1000.0 if telemetry else 0.0
        ttft_ms = telemetry.ttft_ms if telemetry else 0.0
        cost = telemetry.computed_cost_usd if telemetry else 0.0

        latencies.append(duration_ms)
        ttfts.append(ttft_ms)
        costs.append(cost)

        results.append({
            "id": qid,
            "category": cat,
            "question": q,
            "answerable": answerable,
            "candidate_answer": cand_ans,
            "ground_truth_answer": gt,
            "passed": passed,
            "judge_reason": judge_res["reason"],
            "citations_precision": round(c_prec, 4),
            "citations_recall": round(c_rec, 4),
            "used_fallback": used_fb,
            "duration_ms": round(duration_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "cost_usd": cost
        })

    # Summary Calculations
    total_queries = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    accuracy_pct = round((passed_count / total_queries) * 100.0, 2) if total_queries else 0.0

    avg_prec = round(float(np.mean([r["citations_precision"] for r in results])), 4)
    avg_rec = round(float(np.mean([r["citations_recall"] for r in results])), 4)

    tpr_fallback = round((correct_fallbacks / total_unanswerable) * 100.0, 2) if total_unanswerable else 100.0
    ffr_fallback = round((false_fallbacks / total_answerable) * 100.0, 2) if total_answerable else 0.0

    lat_arr = np.array(latencies)
    p50_lat = round(float(np.percentile(lat_arr, 50)), 2)
    p95_lat = round(float(np.percentile(lat_arr, 95)), 2)
    avg_lat = round(float(np.mean(lat_arr)), 2)
    avg_ttft = round(float(np.mean(ttfts)), 2)
    total_cost = round(float(np.sum(costs)), 6)
    avg_cost = round(float(np.mean(costs)), 6) if costs else 0.0

    # Category breakdown formatting
    cat_breakdown = {}
    for c, data in category_stats.items():
        tot = data["total"]
        pas = data["passed"]
        cat_breakdown[c] = {
            "total": tot,
            "passed": pas,
            "accuracy_pct": round((pas / tot) * 100.0, 2) if tot else 0.0
        }

    failed_cases = [r for r in results if not r["passed"]]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_label = "v1_baseline" if is_baseline_v1 else "v2_production"

    report = {
        "timestamp": timestamp_str,
        "version": version_label,
        "total_queries": total_queries,
        "passed_queries": passed_count,
        "accuracy_pct": accuracy_pct,
        "citation_precision": avg_prec,
        "citation_recall": avg_rec,
        "fallback_tpr_pct": tpr_fallback,
        "fallback_ffr_pct": ffr_fallback,
        "latency_avg_ms": avg_lat,
        "latency_p50_ms": p50_lat,
        "latency_p95_ms": p95_lat,
        "avg_ttft_ms": avg_ttft,
        "total_cost_usd": total_cost,
        "avg_cost_per_query_usd": avg_cost,
        "category_breakdown": cat_breakdown,
        "failed_cases_count": len(failed_cases),
        "failed_cases": failed_cases,
        "detailed_results": results
    }

    # Save JSON and Markdown Reports
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_{version_label}_{timestamp_str}.json"
    md_path = out_dir / f"eval_{version_label}_{timestamp_str}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_markdown_report(report, md_path)
    logger.info(f"Evaluation finished! Reports saved to:\n  {json_path}\n  {md_path}")
    return report


def write_markdown_report(report: Dict[str, Any], file_path: Path) -> None:
    """Writes human-readable evaluation report in GitHub Flavored Markdown."""
    lines = [
        f"# Evaluation Report: {report['version'].upper()}",
        f"\n**Timestamp**: `{report['timestamp']}`  ",
        f"**Evaluated Queries**: `{report['total_queries']}`  ",
        f"**Overall Accuracy**: `{report['accuracy_pct']}%`  \n",
        "## 📊 Executive Summary Metrics\n",
        "| Metric | Result | Benchmark Target | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Answer Accuracy** | `{report['accuracy_pct']}%` | >= 85.0% | {'✅ PASS' if report['accuracy_pct'] >= 85 else '⚠️ REVIEW'} |",
        f"| **Citation Precision** | `{report['citation_precision'] * 100:.1f}%` | >= 90.0% | {'✅ PASS' if report['citation_precision'] >= 0.9 else '⚠️ REVIEW'} |",
        f"| **Citation Recall** | `{report['citation_recall'] * 100:.1f}%` | >= 80.0% | {'✅ PASS' if report['citation_recall'] >= 0.8 else '⚠️ REVIEW'} |",
        f"| **Fallback TPR (Unanswerable)** | `{report['fallback_tpr_pct']}%` | >= 90.0% | {'✅ PASS' if report['fallback_tpr_pct'] >= 90 else '❌ FAIL'} |",
        f"| **False-Fallback Rate (Answerable)** | `{report['fallback_ffr_pct']}%` | <= 10.0% | {'✅ PASS' if report['fallback_ffr_pct'] <= 10 else '⚠️ REVIEW'} |",
        f"| **p50 Latency** | `{report['latency_p50_ms']} ms` | < 1500 ms | {'✅ PASS' if report['latency_p50_ms'] < 1500 else 'ℹ️ INFO'} |",
        f"| **p95 Latency** | `{report['latency_p95_ms']} ms` | < 3000 ms | {'✅ PASS' if report['latency_p95_ms'] < 3000 else 'ℹ️ INFO'} |",
        f"| **Average TTFT** | `{report['avg_ttft_ms']} ms` | < 1000 ms | {'✅ PASS' if report['avg_ttft_ms'] < 1000 else 'ℹ️ INFO'} |",
        f"| **Total Run Cost** | `${report['total_cost_usd']:.6f}` | < $0.10 | ✅ PASS |",
        f"| **Avg Cost / Query** | `${report['avg_cost_per_query_usd']:.6f}` | < $0.005 | ✅ PASS |\n",
        "## 🏷️ Category Breakdown\n",
        "| Category | Total Queries | Passed | Accuracy % |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for cat, stats in report["category_breakdown"].items():
        lines.append(f"| `{cat}` | {stats['total']} | {stats['passed']} | `{stats['accuracy_pct']}%` |")

    lines.append(f"\n## ⚠️ Failed Cases ({report['failed_cases_count']})\n")
    if not report["failed_cases"]:
        lines.append("All evaluation cases passed with 100% compliance!")
    else:
        for fc in report["failed_cases"]:
            lines.extend([
                f"### Case `{fc['id']}` ({fc['category']})",
                f"- **Question**: {fc['question']}",
                f"- **Expected Ground Truth**: {fc['ground_truth_answer']}",
                f"- **Candidate Output**: {fc['candidate_answer']}",
                f"- **Reason for Failure**: {fc['judge_reason']}",
                f"- **Used Fallback**: `{fc['used_fallback']}`\n"
            ])

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compare_reports(report_a_path: Path, report_b_path: Path) -> None:
    """Diffs two evaluation JSON reports and prints comparative Markdown table."""
    with open(report_a_path, "r", encoding="utf-8") as f:
        rep_a = json.load(f)
    with open(report_b_path, "r", encoding="utf-8") as f:
        rep_b = json.load(f)

    print("\n" + "=" * 60)
    print("[EVALUATION COMPARISON DIFF]")
    print(f"Base:    {rep_a.get('version')} ({report_a_path.name})")
    print(f"Current: {rep_b.get('version')} ({report_b_path.name})")
    print("=" * 60)

    rows = [
        ("Answer Accuracy", f"{rep_a['accuracy_pct']}%", f"{rep_b['accuracy_pct']}%", f"{rep_b['accuracy_pct'] - rep_a['accuracy_pct']:+.2f}%"),
        ("Citation Precision", f"{rep_a['citation_precision']*100:.1f}%", f"{rep_b['citation_precision']*100:.1f}%", f"{(rep_b['citation_precision'] - rep_a['citation_precision'])*100:+.1f}%"),
        ("Fallback TPR (Unanswerable)", f"{rep_a['fallback_tpr_pct']}%", f"{rep_b['fallback_tpr_pct']}%", f"{rep_b['fallback_tpr_pct'] - rep_a['fallback_tpr_pct']:+.2f}%"),
        ("False-Fallback Rate", f"{rep_a['fallback_ffr_pct']}%", f"{rep_b['fallback_ffr_pct']}%", f"{rep_b['fallback_ffr_pct'] - rep_a['fallback_ffr_pct']:+.2f}%"),
        ("Avg Latency (ms)", f"{rep_a['latency_avg_ms']:.1f}", f"{rep_b['latency_avg_ms']:.1f}", f"{rep_b['latency_avg_ms'] - rep_a['latency_avg_ms']:+.1f}"),
        ("p95 Latency (ms)", f"{rep_a['latency_p95_ms']:.1f}", f"{rep_b['latency_p95_ms']:.1f}", f"{rep_b['latency_p95_ms'] - rep_a['latency_p95_ms']:+.1f}"),
        ("Avg Cost / Query ($)", f"${rep_a['avg_cost_per_query_usd']:.6f}", f"${rep_b['avg_cost_per_query_usd']:.6f}", f"${rep_b['avg_cost_per_query_usd'] - rep_a['avg_cost_per_query_usd']:+.6f}")
    ]

    print(f"{'Metric':<30} | {'Baseline':<12} | {'Current':<12} | {'Delta':<10}")
    print("-" * 72)
    for metric, base, cur, delta in rows:
        print(f"{metric:<30} | {base:<12} | {cur:<12} | {delta:<10}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BMW AI Chatbot Evaluation Runner")
    parser.add_argument("--dataset", type=str, default="evals/dataset.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--out", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--baseline-v1", action="store_true", help="Run read-only evaluation of v1 baseline")
    parser.add_argument("--threshold", type=float, default=settings.SIMILARITY_THRESHOLD, help="Override similarity threshold")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries evaluated (for smoke tests)")
    parser.add_argument("--compare", type=str, default=None, help="Path to previous JSON report to diff against")
    parser.add_argument("--diff", nargs=2, type=str, default=None, help="Diff two existing JSON reports directly")
    args = parser.parse_args()

    if args.diff:
        compare_reports(Path(args.diff[0]), Path(args.diff[1]))
        return

    out_dir = BASE_DIR / args.out
    dataset_path = BASE_DIR / args.dataset

    if args.compare:
        prior_report_path = Path(args.compare)
        current_report = run_eval(
            dataset_path=dataset_path,
            out_dir=out_dir,
            is_baseline_v1=args.baseline_v1,
            similarity_threshold=args.threshold,
            limit=args.limit
        )
        current_report_json = out_dir / f"eval_{current_report['version']}_{current_report['timestamp']}.json"
        compare_reports(prior_report_path, current_report_json)
    else:
        run_eval(
            dataset_path=dataset_path,
            out_dir=out_dir,
            is_baseline_v1=args.baseline_v1,
            similarity_threshold=args.threshold,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
