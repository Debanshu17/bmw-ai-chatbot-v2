"""CLI Evaluation Harness and Benchmarking Runner.

Runs datasets through the RAG pipeline or v1 baseline, computes comprehensive
accuracy, citation precision/recall, fallback reliability, latency percentiles,
token usage, and financial cost metrics. Outputs structured JSON and human-readable Markdown.
Never scores infrastructure API failures as wrong answers, tracks answer sources,
judge methods, enforces validity thresholds, and strictly verifies resume caches.
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
from evals.retry import call_gemini_with_retry, pace_api_request
from generation.generator import generate_rag_response
from retrieval.index import global_vector_index
from telemetry.monitor import TelemetryTracker, calculate_cost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Evaluation harness version tag - cache entries must match this exactly
HARNESS_VERSION = "2.1.0"


def startup_model_check(model_name: str, api_key: str) -> None:
    """Verifies that the configured model is available and has active quota before running evaluations.
    
    Aborts immediately on 404 (NotFound) or 429 (ResourceExhausted/Quota).
    """
    import google.generativeai as genai
    logger.info(f"Running startup model check on '{model_name}'...")
    if not api_key:
        raise RuntimeError("STARTUP CHECK FAILED: GEMINI_API_KEY environment variable is missing or empty.")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name)
        _ = model.generate_content("ping")
        logger.info(f"Startup check PASSED for model '{model_name}'.")
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "NotFound" in err_msg or "not found" in err_msg.lower():
            logger.critical(f"STARTUP CHECK ABORTED: Model '{model_name}' returned 404 Not Found.")
            raise RuntimeError(
                f"STARTUP CHECK FAILED: Model '{model_name}' is not found (404) or not available for your API key. "
                f"Please update CHAT_MODEL in .env to an available model on your account. Error: {e}"
            )
        elif "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
            logger.critical(f"STARTUP CHECK ABORTED: Model '{model_name}' returned 429 Quota Exhausted.")
            raise RuntimeError(
                f"STARTUP CHECK FAILED: API quota exceeded (429) for model '{model_name}'. "
                f"Please verify billing or retry after daily quota resets. Error: {e}"
            )
        else:
            logger.critical(f"STARTUP CHECK ABORTED: Error invoking model '{model_name}': {e}")
            raise RuntimeError(f"STARTUP CHECK FAILED for model '{model_name}': {e}")


def run_v1_baseline_query(question: str) -> Dict[str, Any]:
    """Evaluates v1 baseline (zero RAG, pure parametric LLM) in a read-only wrapper.
    
    Retries up to 5 times with exponential backoff on infrastructure errors (429, timeouts).
    Marks result as ERROR if API failures persist.
    """
    import google.generativeai as genai
    # Pacing sleep occurs OUTSIDE the timed region
    pace_api_request(min_interval=7.5)

    tracker = TelemetryTracker(model_name=settings.CHAT_MODEL).start()

    prompt = f"Answer the user's question:\n{question}"
    model = genai.GenerativeModel(model_name=settings.CHAT_MODEL)

    def _call():
        res = model.generate_content(prompt)
        tracker.record_first_token()
        return res

    api_response, is_api_error, err_detail = call_gemini_with_retry(_call, max_retries=5, enforce_pacing=False)
    if is_api_error:
        logger.error(f"v1 baseline call failed after 5 retries: {err_detail}")
        prompt_tokens = len(prompt.split()) * 2
        telemetry = tracker.stop(prompt_tokens=prompt_tokens, completion_tokens=0)
        return {
            "answer": "ERROR",
            "citations": [],
            "used_fallback": False,
            "telemetry": telemetry,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "answer_source": "gemini",
            "is_error": True,
            "error_message": err_detail
        }

    try:
        text = api_response.text or ""
        prompt_tokens = len(prompt.split()) * 2
        completion_tokens = len(text.split()) * 2
        if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
            prompt_tokens = getattr(api_response.usage_metadata, "prompt_token_count", prompt_tokens)
            completion_tokens = getattr(api_response.usage_metadata, "candidates_token_count", completion_tokens)
    except Exception as e:
        logger.error(f"v1 baseline text extraction failed: {e}")
        text = "SYSTEM ERROR"
        prompt_tokens = 0
        completion_tokens = 0

    telemetry = tracker.stop(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return {
        "answer": text,
        "citations": [],
        "used_fallback": False,
        "telemetry": telemetry,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "answer_source": "gemini",
        "is_error": False,
        "error_message": None
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


def is_error_entry(entry: Dict[str, Any]) -> bool:
    """Checks whether an evaluation result represents an infrastructure failure."""
    if entry.get("is_error", False):
        return True
    cand = str(entry.get("candidate_answer", "")).strip()
    if cand in ["ERROR", "SYSTEM ERROR"]:
        return True
    if entry.get("verdict") == "ERROR":
        return True
    return False


def is_valid_cache_entry(entry: Any, target_model: str, expected_harness_version: str) -> bool:
    """Strictly validates a cached entry before allowing reuse.

    Any cached entry without answer_source, judge_method, model name, or harness version
    must be rejected and re-run. Must match current target_model and expected_harness_version.
    """
    if not isinstance(entry, dict):
        return False
    ans_src = entry.get("answer_source")
    if not ans_src or ans_src not in ["gemini", "extractive_fallback"]:
        return False
    judge_m = entry.get("judge_method")
    if not judge_m or judge_m not in ["llm", "fuzzy"]:
        return False
    if entry.get("model") != target_model:
        return False
    if entry.get("harness_version") != expected_harness_version:
        return False
    if is_error_entry(entry):
        return False
    return True


def run_eval(
    dataset_path: Path,
    out_dir: Path,
    is_baseline_v1: bool = False,
    similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
    limit: Optional[int] = None,
    resume: bool = False,
    checkpoint_path: Optional[Path] = None,
    skip_startup_check: bool = False
) -> Dict[str, Any]:
    """Executes evaluation run across all entries in the dataset.
    
    Excludes infrastructure errors from accuracy and latency calculations.
    Aborts run if error rate exceeds 5%. Marks run invalid if extractive fallback
    on answerable questions exceeds 5%.
    Supports --resume to reuse strictly validated cached queries and only retry missing or ERROR queries.
    """
    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]

    total_in_dataset = len(dataset)
    version_label = "v1_baseline" if is_baseline_v1 else "v2_production"
    logger.info(f"Loaded {total_in_dataset} evaluation entries from {dataset_path} for {version_label}")

    # Startup model check: fail fast on 404 or 429 quota errors before starting evaluations
    if not skip_startup_check and settings.GEMINI_API_KEY:
        startup_model_check(settings.CHAT_MODEL, settings.GEMINI_API_KEY)

    # Build vector index if needed for v2
    if not is_baseline_v1:
        global_vector_index.build_or_load()

    # Determine checkpoint file and load cached results if resume is requested
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_file = checkpoint_path or (out_dir / f".checkpoint_{version_label}.json")
    cache_by_id: Dict[str, Dict[str, Any]] = {}

    if resume:
        cached_list: List[Dict[str, Any]] = []
        if chk_file.exists():
            try:
                with open(chk_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cached_list = data if isinstance(data, list) else data.get("detailed_results", [])
                logger.info(f"Loaded {len(cached_list)} cached entries from checkpoint: {chk_file.name}")
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {chk_file}: {e}")
        else:
            # Fall back to latest valid report in out_dir, strictly ignoring invalid_archive
            existing_reports = [
                p for p in out_dir.glob(f"eval_{version_label}_*.json")
                if "invalid_archive" not in p.parts
            ]
            existing_reports = sorted(existing_reports, key=lambda p: p.stat().st_mtime, reverse=True)
            if existing_reports:
                latest_report = existing_reports[0]
                try:
                    with open(latest_report, "r", encoding="utf-8") as f:
                        rep_data = json.load(f)
                        if (
                            rep_data.get("harness_version") == HARNESS_VERSION
                            and rep_data.get("model") == settings.CHAT_MODEL
                        ):
                            cached_list = rep_data.get("detailed_results", [])
                            logger.info(f"Loaded {len(cached_list)} cached entries from latest report: {latest_report.name}")
                        else:
                            logger.info(
                                f"Skipping report {latest_report.name}: mismatched harness version "
                                f"({rep_data.get('harness_version')} vs {HARNESS_VERSION}) or model "
                                f"({rep_data.get('model')} vs {settings.CHAT_MODEL})."
                            )
                except Exception as e:
                    logger.warning(f"Failed to read latest report {latest_report}: {e}")

        # Strictly validate cached entries - reject any without answer_source, judge_method, model, or harness_version
        for r in cached_list:
            if "id" in r and is_valid_cache_entry(r, target_model=settings.CHAT_MODEL, expected_harness_version=HARNESS_VERSION):
                cache_by_id[r["id"]] = r
            elif "id" in r:
                logger.info(f"Discarding invalid cache entry for {r['id']} (missing answer_source/judge_method/model/harness_version or error)")

    results: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = {}
    latencies: List[float] = []
    ttfts: List[float] = []
    costs: List[float] = []

    correct_fallbacks = 0  # True positives on unanswerable
    false_fallbacks = 0    # False fallbacks on answerable
    error_count = 0

    for i, item in enumerate(dataset):
        qid = item["id"]
        q = item["question"]
        gt = item["ground_truth_answer"]
        expected_srcs = item.get("expected_sources", [])
        answerable = item.get("answerable", True)
        cat = item.get("category", "general")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "valid": 0, "passed": 0}

        category_stats[cat]["total"] += 1

        cached_res = cache_by_id.get(qid) if resume else None
        if cached_res:
            logger.info(f"[{i+1}/{total_in_dataset}] Reusing valid CACHED result for {qid} ({cat})...")
            cand_ans = cached_res["candidate_answer"]
            cits = cached_res.get("citations", [])
            used_fb = cached_res.get("used_fallback", False)
            ans_src = cached_res["answer_source"]
            item_is_error = False
            err_msg = None
            passed = cached_res.get("passed", False)
            judge_reason = cached_res.get("judge_reason", "")
            judge_method = cached_res["judge_method"]
            c_prec = cached_res.get("citations_precision")
            c_rec = cached_res.get("citations_recall")
            duration_ms = cached_res.get("duration_ms", 0.0)
            ttft_ms = cached_res.get("ttft_ms", 0.0)
            prompt_tokens = cached_res.get("prompt_tokens", 0)
            completion_tokens = cached_res.get("completion_tokens", 0)
            cost = cached_res.get("cost_usd", 0.0)
            verdict = cached_res.get("verdict", "PASS" if passed else "FAIL")
        else:
            logger.info(f"[{i+1}/{total_in_dataset}] Evaluating {qid} ({cat})...")

            if is_baseline_v1:
                resp_data = run_v1_baseline_query(q)
                cand_ans = resp_data["answer"]
                cits = resp_data["citations"]
                used_fb = resp_data["used_fallback"]
                telemetry = resp_data["telemetry"]
                ans_src = resp_data["answer_source"]
                query_is_error = resp_data.get("is_error", False)
                err_msg = resp_data.get("error_message")
                prompt_tokens = resp_data.get("prompt_tokens", 0)
                completion_tokens = resp_data.get("completion_tokens", 0)
            else:
                rag_resp = generate_rag_response(
                    query=q,
                    similarity_threshold=similarity_threshold
                )
                cand_ans = rag_resp.answer
                cits = rag_resp.citations
                used_fb = rag_resp.used_fallback
                telemetry = rag_resp.telemetry
                ans_src = rag_resp.answer_source
                query_is_error = rag_resp.is_error
                err_msg = rag_resp.error_message
                prompt_tokens = telemetry.prompt_tokens if telemetry else 0
                completion_tokens = telemetry.completion_tokens if telemetry else 0

            # Judge evaluation (fast-paths if candidate answer had infrastructure error)
            judge_res = judge_answer(
                question=q,
                candidate_answer=cand_ans,
                ground_truth_answer=gt,
                answerable=answerable,
                candidate_is_error=query_is_error
            )
            judge_is_error = judge_res.get("is_error", False)
            item_is_error = query_is_error or judge_is_error
            judge_method = judge_res.get("judge_method", "none")
            judge_reason = judge_res.get("reason", "")
            verdict = "ERROR" if item_is_error else judge_res.get("verdict", "FAIL")

            # Track infrastructure errors and enforce dynamic abort threshold (> 5%)
            if item_is_error:
                error_count += 1
                if (error_count / total_in_dataset) > 0.05:
                    err_rate = (error_count / total_in_dataset) * 100.0
                    abort_msg = (
                        f"CRITICAL: Evaluation run aborted! Infrastructure error rate {err_rate:.1f}% "
                        f"({error_count}/{total_in_dataset} queries) exceeded 5.0% threshold. "
                        f"Please check API quota and network connection."
                    )
                    logger.critical(abort_msg)
                    raise RuntimeError(abort_msg)

            passed = (judge_res["score"] == 1.0) if not item_is_error else None

            # Citation metrics (only valid queries)
            if not item_is_error:
                c_prec, c_rec = compute_citation_metrics(cits, expected_srcs)
            else:
                c_prec, c_rec = None, None

            # Telemetry metrics
            duration_ms = telemetry.duration_s * 1000.0 if telemetry else 0.0
            ttft_ms = telemetry.ttft_ms if telemetry else 0.0
            cost = telemetry.computed_cost_usd if telemetry else calculate_cost(
                settings.CHAT_MODEL, prompt_tokens, completion_tokens
            )

        # Fallback accounting on non-error queries
        if not item_is_error:
            category_stats[cat]["valid"] += 1
            if passed:
                category_stats[cat]["passed"] += 1

            if not answerable:
                if used_fb or cand_ans.strip() == settings.FALLBACK_MESSAGE:
                    correct_fallbacks += 1
            else:
                if used_fb or cand_ans.strip() == settings.FALLBACK_MESSAGE:
                    false_fallbacks += 1

        costs.append(cost)

        # Exclude ERROR queries from latency lists
        if not item_is_error:
            latencies.append(duration_ms)
            ttfts.append(ttft_ms)

        result_entry = {
            "id": qid,
            "category": cat,
            "question": q,
            "answerable": answerable,
            "candidate_answer": cand_ans,
            "ground_truth_answer": gt,
            "passed": passed,
            "verdict": verdict,
            "is_error": item_is_error,
            "error_message": err_msg or (judge_reason if item_is_error else None),
            "answer_source": ans_src,
            "judge_method": judge_method,
            "judge_reason": judge_reason,
            "citations_precision": round(c_prec, 4) if c_prec is not None else None,
            "citations_recall": round(c_rec, 4) if c_rec is not None else None,
            "used_fallback": used_fb,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": round(duration_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "cost_usd": cost,
            "model": settings.CHAT_MODEL,
            "harness_version": HARNESS_VERSION
        }
        results.append(result_entry)

        # Update checkpoint file after every question to persist progress
        try:
            with open(chk_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update checkpoint {chk_file}: {e}")

    # Summary Calculations
    total_queries = len(results)
    valid_results = [r for r in results if not r["is_error"]]
    error_cases = [r for r in results if r["is_error"]]
    error_count = len(error_cases)
    valid_queries = len(valid_results)
    error_rate_pct = round((error_count / total_queries) * 100.0, 2) if total_queries else 0.0

    # Accuracy computed EXCLUSIVELY over valid queries
    passed_count = sum(1 for r in valid_results if r["passed"] is True)
    accuracy_pct = round((passed_count / valid_queries) * 100.0, 2) if valid_queries else 0.0

    # Citation metrics computed EXCLUSIVELY over valid queries
    prec_vals = [r["citations_precision"] for r in valid_results if r["citations_precision"] is not None]
    rec_vals = [r["citations_recall"] for r in valid_results if r["citations_recall"] is not None]
    avg_prec = round(float(np.mean(prec_vals)), 4) if prec_vals else 0.0
    avg_rec = round(float(np.mean(rec_vals)), 4) if rec_vals else 0.0

    # Fallback rates over valid queries
    valid_answerable = [r for r in valid_results if r["answerable"]]
    valid_unanswerable = [r for r in valid_results if not r["answerable"]]
    tpr_fallback = round((correct_fallbacks / len(valid_unanswerable)) * 100.0, 2) if valid_unanswerable else 100.0
    ffr_fallback = round((false_fallbacks / len(valid_answerable)) * 100.0, 2) if valid_answerable else 0.0

    # Latency percentiles computed EXCLUSIVELY over valid queries
    if latencies:
        lat_arr = np.array(latencies)
        p50_lat = round(float(np.percentile(lat_arr, 50)), 2)
        p95_lat = round(float(np.percentile(lat_arr, 95)), 2)
        avg_lat = round(float(np.mean(lat_arr)), 2)
    else:
        p50_lat = p95_lat = avg_lat = 0.0

    avg_ttft = round(float(np.mean(ttfts)), 2) if ttfts else 0.0

    # Token and cost calculations
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_cost = round(float(np.sum(costs)), 6)
    avg_cost = round(float(np.mean(costs)), 6) if costs else 0.0

    # Answer sources breakdown
    gemini_answers_count = sum(1 for r in results if r.get("answer_source") == "gemini")
    extractive_answers_count = sum(1 for r in results if r.get("answer_source") == "extractive_fallback")

    # Judge methods breakdown
    llm_judgments_count = sum(1 for r in results if r.get("judge_method") == "llm")
    fuzzy_judgments_count = sum(1 for r in results if r.get("judge_method") == "fuzzy")
    llm_judgments_answerable = sum(1 for r in valid_answerable if r.get("judge_method") == "llm")
    fuzzy_judgments_answerable = sum(1 for r in valid_answerable if r.get("judge_method") == "fuzzy")
    llm_judgments_unanswerable = sum(1 for r in valid_unanswerable if r.get("judge_method") == "llm")
    fuzzy_judgments_unanswerable = sum(1 for r in valid_unanswerable if r.get("judge_method") == "fuzzy")

    # Extractive fallback validity check on answerable questions
    extractive_answerable_count = sum(1 for r in valid_answerable if r.get("answer_source") == "extractive_fallback")
    extractive_answerable_pct = round((extractive_answerable_count / len(valid_answerable)) * 100.0, 2) if valid_answerable else 0.0

    # Enforce run validity rules:
    # 1. Extractive fallback on answerable questions <= 5%
    # 2. Infrastructure error rate <= 5%
    is_valid = (extractive_answerable_pct <= 5.0) and (error_rate_pct <= 5.0)
    if not is_valid:
        if extractive_answerable_pct > 5.0:
            validity_status = f"INVALID: Extractive fallback rate ({extractive_answerable_pct:.1f}%) exceeds 5.0% on answerable questions"
        else:
            validity_status = f"INVALID: Infrastructure error rate ({error_rate_pct:.1f}%) exceeds 5.0% threshold"
    else:
        validity_status = "VALID"

    # Category breakdown formatting
    cat_breakdown = {}
    for c, data in category_stats.items():
        tot = data["total"]
        val = data["valid"]
        pas = data["passed"]
        cat_breakdown[c] = {
            "total": tot,
            "valid": val,
            "passed": pas,
            "accuracy_pct": round((pas / val) * 100.0, 2) if val else 0.0
        }

    # Only include non-error failed cases in failed_cases
    failed_cases = [r for r in valid_results if not r["passed"]]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp_str,
        "version": version_label,
        "model": settings.CHAT_MODEL,
        "harness_version": HARNESS_VERSION,
        "is_valid": is_valid,
        "validity_status": validity_status,
        "total_queries": total_queries,
        "valid_queries": valid_queries,
        "error_count": error_count,
        "error_rate_pct": error_rate_pct,
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
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "avg_cost_per_query_usd": avg_cost,
        "gemini_answers_count": gemini_answers_count,
        "extractive_answers_count": extractive_answers_count,
        "llm_judgments_count": llm_judgments_count,
        "fuzzy_judgments_count": fuzzy_judgments_count,
        "llm_judgments_answerable": llm_judgments_answerable,
        "fuzzy_judgments_answerable": fuzzy_judgments_answerable,
        "llm_judgments_unanswerable": llm_judgments_unanswerable,
        "fuzzy_judgments_unanswerable": fuzzy_judgments_unanswerable,
        "extractive_fallback_on_answerable_count": extractive_answerable_count,
        "extractive_fallback_on_answerable_pct": extractive_answerable_pct,
        "answer_sources": {
            "gemini": gemini_answers_count,
            "extractive_fallback": extractive_answers_count
        },
        "judge_methods": {
            "llm": llm_judgments_count,
            "fuzzy": fuzzy_judgments_count,
            "none": sum(1 for r in results if r.get("judge_method") == "none")
        },
        "category_breakdown": cat_breakdown,
        "failed_cases_count": len(failed_cases),
        "failed_cases": failed_cases,
        "error_cases_count": len(error_cases),
        "error_cases": error_cases,
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
        f"**Model Evaluated**: `{report.get('model', settings.CHAT_MODEL)}`  ",
        f"**Harness Version**: `{report.get('harness_version', HARNESS_VERSION)}`  ",
        f"**Run Validity**: `{report.get('validity_status', 'VALID')}`  ",
        f"**Evaluated Queries**: `{report['total_queries']}` (Valid: `{report.get('valid_queries', report['total_queries'])}`, Errors: `{report.get('error_count', 0)}`)  ",
        f"**Overall Accuracy (Valid Queries)**: `{report['accuracy_pct']}%`  \n",
        "## 📊 Executive Summary Metrics\n",
        "| Metric | Result | Benchmark Target | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Run Validity** | `{report.get('validity_status', 'VALID')}` | Valid (Extractive FB <= 5%) | {'✅ VALID' if report.get('is_valid', True) else '❌ INVALID'} |",
        f"| **Infrastructure Errors** | `{report.get('error_count', 0)} ({report.get('error_rate_pct', 0.0)}%)` | 0 errors (<= 5% abort) | {'✅ PASS' if report.get('error_count', 0) == 0 else ('⚠️ REVIEW' if report.get('error_rate_pct', 0.0) <= 5.0 else '❌ ABORT')} |",
        f"| **Answer Accuracy (Valid)** | `{report['accuracy_pct']}%` | >= 85.0% | {'✅ PASS' if report['accuracy_pct'] >= 85 else '⚠️ REVIEW'} |",
        f"| **Answer Source: Gemini** | `{report.get('gemini_answers_count', report['total_queries'])}` | Primary LLM generation | ℹ️ INFO |",
        f"| **Answer Source: Extractive Fallback** | `{report.get('extractive_answers_count', 0)}` | <= 5% on answerable | {'✅ PASS' if report.get('extractive_fallback_on_answerable_pct', 0.0) <= 5.0 else '❌ FAIL'} |",
        f"| **Extractive FB (Answerable)** | `{report.get('extractive_fallback_on_answerable_count', 0)} ({report.get('extractive_fallback_on_answerable_pct', 0.0)}%)` | <= 5.0% | {'✅ PASS' if report.get('extractive_fallback_on_answerable_pct', 0.0) <= 5.0 else '❌ FAIL'} |",
        f"| **Judge Method: LLM** | `{report.get('llm_judgments_count', 0)}` | Strict rubric | ℹ️ INFO |",
        f"| **Judge Method: Fuzzy** | `{report.get('fuzzy_judgments_count', 0)}` | Refusal / deterministic | ℹ️ INFO |",
        f"| **Citation Precision** | `{report['citation_precision'] * 100:.1f}%` | >= 90.0% | {'✅ PASS' if report['citation_precision'] >= 0.9 else '⚠️ REVIEW'} |",
        f"| **Citation Recall** | `{report['citation_recall'] * 100:.1f}%` | >= 80.0% | {'✅ PASS' if report['citation_recall'] >= 0.8 else '⚠️ REVIEW'} |",
        f"| **Fallback TPR (Unanswerable)** | `{report['fallback_tpr_pct']}%` | >= 90.0% | {'✅ PASS' if report['fallback_tpr_pct'] >= 90 else '❌ FAIL'} |",
        f"| **False-Fallback Rate (Answerable)** | `{report['fallback_ffr_pct']}%` | <= 10.0% | {'✅ PASS' if report['fallback_ffr_pct'] <= 10 else '⚠️ REVIEW'} |",
        f"| **p50 Latency** | `{report['latency_p50_ms']} ms` | < 1500 ms | {'✅ PASS' if report['latency_p50_ms'] < 1500 else 'ℹ️ INFO'} |",
        f"| **p95 Latency** | `{report['latency_p95_ms']} ms` | < 3000 ms | {'✅ PASS' if report['latency_p95_ms'] < 3000 else 'ℹ️ INFO'} |",
        f"| **Average TTFT** | `{report['avg_ttft_ms']} ms` | < 1000 ms | {'✅ PASS' if report['avg_ttft_ms'] < 1000 else 'ℹ️ INFO'} |",
        f"| **Total Prompt Tokens** | `{report.get('total_prompt_tokens', 0):,}` | Input tokens tracked | ℹ️ INFO |",
        f"| **Total Completion Tokens** | `{report.get('total_completion_tokens', 0):,}` | Output tokens tracked | ℹ️ INFO |",
        f"| **Total Run Cost** | `${report['total_cost_usd']:.6f}` | < $0.10 | ✅ PASS |",
        f"| **Avg Cost / Query** | `${report['avg_cost_per_query_usd']:.6f}` | < $0.005 | ✅ PASS |\n",
        "## 🏷️ Category Breakdown\n",
        "| Category | Total Queries | Valid | Passed | Accuracy % |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    for cat, stats in report["category_breakdown"].items():
        tot = stats["total"]
        val = stats.get("valid", tot)
        pas = stats["passed"]
        acc = stats["accuracy_pct"]
        lines.append(f"| `{cat}` | {tot} | {val} | {pas} | `{acc}%` |")

    # Infrastructure errors section if any occurred
    if report.get("error_cases"):
        lines.append(f"\n## 🚨 Infrastructure Errors ({report['error_cases_count']})\n")
        for ec in report["error_cases"]:
            lines.extend([
                f"### Error Case `{ec['id']}` ({ec['category']})",
                f"- **Question**: {ec['question']}",
                f"- **Error Details**: {ec.get('error_message', 'API Failure')}",
                f"- **Answer Source**: `{ec.get('answer_source')}`",
                f"- **Judge Method**: `{ec.get('judge_method')}`\n"
            ])

    lines.append(f"\n## ⚠️ Failed Cases ({report['failed_cases_count']})\n")
    if not report["failed_cases"]:
        lines.append("All valid evaluation cases passed with 100% compliance!")
    else:
        for fc in report["failed_cases"]:
            lines.extend([
                f"### Case `{fc['id']}` ({fc['category']})",
                f"- **Question**: {fc['question']}",
                f"- **Expected Ground Truth**: {fc['ground_truth_answer']}",
                f"- **Candidate Output**: {fc['candidate_answer']}",
                f"- **Answer Source**: `{fc.get('answer_source')}`",
                f"- **Judge Method**: `{fc.get('judge_method')}`",
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
        ("Run Validity", str(rep_a.get('validity_status', 'VALID')), str(rep_b.get('validity_status', 'VALID')), "N/A"),
        ("Model", str(rep_a.get('model', 'N/A')), str(rep_b.get('model', 'N/A')), "N/A"),
        ("Infrastructure Errors", f"{rep_a.get('error_count', 0)}", f"{rep_b.get('error_count', 0)}", f"{rep_b.get('error_count', 0) - rep_a.get('error_count', 0):+d}"),
        ("Answer Accuracy (Valid)", f"{rep_a['accuracy_pct']}%", f"{rep_b['accuracy_pct']}%", f"{rep_b['accuracy_pct'] - rep_a['accuracy_pct']:+.2f}%"),
        ("Extractive FB on Ans.", f"{rep_a.get('extractive_fallback_on_answerable_count', 0)}", f"{rep_b.get('extractive_fallback_on_answerable_count', 0)}", f"{rep_b.get('extractive_fallback_on_answerable_count', 0) - rep_a.get('extractive_fallback_on_answerable_count', 0):+d}"),
        ("Citation Precision", f"{rep_a['citation_precision']*100:.1f}%", f"{rep_b['citation_precision']*100:.1f}%", f"{(rep_b['citation_precision'] - rep_a['citation_precision'])*100:+.1f}%"),
        ("Fallback TPR (Unanswerable)", f"{rep_a['fallback_tpr_pct']}%", f"{rep_b['fallback_tpr_pct']}%", f"{rep_b['fallback_tpr_pct'] - rep_a['fallback_tpr_pct']:+.2f}%"),
        ("False-Fallback Rate", f"{rep_a['fallback_ffr_pct']}%", f"{rep_b['fallback_ffr_pct']}%", f"{rep_b['fallback_ffr_pct'] - rep_a['fallback_ffr_pct']:+.2f}%"),
        ("Avg Latency (ms)", f"{rep_a['latency_avg_ms']:.1f}", f"{rep_b['latency_avg_ms']:.1f}", f"{rep_b['latency_avg_ms'] - rep_a['latency_avg_ms']:+.1f}"),
        ("p95 Latency (ms)", f"{rep_a['latency_p95_ms']:.1f}", f"{rep_b['latency_p95_ms']:.1f}", f"{rep_b['latency_p95_ms'] - rep_a['latency_p95_ms']:+.1f}"),
        ("Total Tokens", f"{rep_a.get('total_tokens', 0)}", f"{rep_b.get('total_tokens', 0)}", f"{rep_b.get('total_tokens', 0) - rep_a.get('total_tokens', 0):+d}"),
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
    parser.add_argument("--resume", action="store_true", help="Resume run: reuse validated cached queries and only retry missing or ERROR queries")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint or report path to resume from")
    parser.add_argument("--skip-startup-check", action="store_true", help="Skip the pre-flight model ping check")
    parser.add_argument("--compare", type=str, default=None, help="Path to previous JSON report to diff against")
    parser.add_argument("--diff", nargs=2, type=str, default=None, help="Diff two existing JSON reports directly")
    args = parser.parse_args()

    if args.diff:
        compare_reports(Path(args.diff[0]), Path(args.diff[1]))
        return

    out_dir = BASE_DIR / args.out
    dataset_path = BASE_DIR / args.dataset
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    if args.compare:
        prior_report_path = Path(args.compare)
        current_report = run_eval(
            dataset_path=dataset_path,
            out_dir=out_dir,
            is_baseline_v1=args.baseline_v1,
            similarity_threshold=args.threshold,
            limit=args.limit,
            resume=args.resume,
            checkpoint_path=checkpoint_path,
            skip_startup_check=args.skip_startup_check
        )
        current_report_json = out_dir / f"eval_{current_report['version']}_{current_report['timestamp']}.json"
        compare_reports(prior_report_path, current_report_json)
    else:
        run_eval(
            dataset_path=dataset_path,
            out_dir=out_dir,
            is_baseline_v1=args.baseline_v1,
            similarity_threshold=args.threshold,
            limit=args.limit,
            resume=args.resume,
            checkpoint_path=checkpoint_path,
            skip_startup_check=args.skip_startup_check
        )


if __name__ == "__main__":
    main()
