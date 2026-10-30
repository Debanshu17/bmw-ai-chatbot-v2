"""Unit tests for evaluation harness error handling, retry backoff, and validity checks."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import settings
from evals.judge import judge_answer
from evals.retry import call_gemini_with_retry, is_api_failure
from evals.run_evals import HARNESS_VERSION, is_valid_cache_entry, run_eval, startup_model_check
from generation.schemas import Citation, RAGResponse
from telemetry.monitor import TelemetryData


def test_is_api_failure_detection():
    assert is_api_failure(Exception("429 Resource has been exhausted (e.g. check quota)."))
    assert is_api_failure(Exception("503 Service Unavailable"))
    assert is_api_failure(Exception("Deadline exceeded during call"))
    assert is_api_failure(Exception("rate limit exceeded"))
    assert not is_api_failure(ValueError("Invalid syntax"))
    assert not is_api_failure(KeyError("missing_field"))


def test_call_gemini_with_retry_success_after_retries():
    attempts = 0

    def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("429 Resource has been exhausted")
        return "success_response"

    res, is_error, err_msg = call_gemini_with_retry(
        flaky_func, max_retries=5, initial_backoff=0.01, backoff_factor=1.5
    )
    assert attempts == 3
    assert is_error is False
    assert res == "success_response"
    assert err_msg is None


def test_call_gemini_with_retry_failure_after_max_retries():
    attempts = 0

    def constantly_failing_func():
        nonlocal attempts
        attempts += 1
        raise Exception("429 Resource has been exhausted")

    res, is_error, err_msg = call_gemini_with_retry(
        constantly_failing_func, max_retries=5, initial_backoff=0.01, backoff_factor=1.5
    )
    # 1 initial try + 5 retries = 6 total attempts
    assert attempts == 6
    assert is_error is True
    assert res is None
    assert "429" in err_msg


def test_judge_fast_path_on_error():
    # Never grade infrastructure error as wrong answer (verdict must be ERROR, score None)
    res = judge_answer(
        question="What is the tire pressure?",
        candidate_answer="ERROR",
        ground_truth_answer="32 psi",
        answerable=True,
        candidate_is_error=True
    )
    assert res["verdict"] == "ERROR"
    assert res["score"] is None
    assert res["is_error"] is True
    assert res["judge_method"] == "none"


def test_judge_method_recording():
    # Test fuzzy match path when API key is not present or LLM call is mocked
    with patch("evals.judge.settings.GEMINI_API_KEY", ""):
        res = judge_answer(
            question="What is the oil spec?",
            candidate_answer="Use BMW Longlife-01 FE 0W-30 oil.",
            ground_truth_answer="Use BMW Longlife-01 FE 0W-30 synthetic oil.",
            answerable=True,
            candidate_is_error=False
        )
        assert res["judge_method"] == "fuzzy"
        assert res["is_error"] is False


def test_eval_excludes_error_from_accuracy_and_latency(tmp_path: Path):
    # Create mini dataset with 20 items (so 1 error = 5% exactly, no abort)
    dataset_file = tmp_path / "test_dataset.jsonl"
    entries = []
    for i in range(20):
        entries.append({
            "id": f"Q{i+1:02d}",
            "question": f"Question {i+1}?",
            "ground_truth_answer": "Expected Answer",
            "expected_sources": [{"doc": "manual.pdf", "page_or_chapter": "10"}],
            "answerable": True,
            "category": "test"
        })
    with open(dataset_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    def mock_generate_rag(query: str, **kwargs):
        # First query encounters an infrastructure error
        if "Question 1?" in query:
            return RAGResponse(
                answer="ERROR",
                citations=[],
                confidence=0.0,
                used_fallback=True,
                telemetry=TelemetryData(duration_s=5.0, ttft_ms=5000.0, computed_cost_usd=0.0),
                answer_source="gemini",
                is_error=True,
                error_message="429 ResourceExhausted"
            )
        # Remaining 19 queries succeed with valid answers
        return RAGResponse(
            answer="Expected Answer",
            citations=[Citation(chunk_id="c1", doc="manual.pdf", page_or_chapter="10", quote_snippet="snippet")],
            confidence=0.9,
            used_fallback=False,
            telemetry=TelemetryData(duration_s=1.0, ttft_ms=200.0, computed_cost_usd=0.0001),
            answer_source="gemini",
            is_error=False
        )

    with patch("evals.run_evals.generate_rag_response", side_effect=mock_generate_rag), \
         patch("evals.run_evals.judge_answer") as mock_judge:
        
        def mock_judge_impl(question, candidate_answer, ground_truth_answer, answerable, candidate_is_error=False):
            if candidate_is_error or candidate_answer == "ERROR":
                return {"verdict": "ERROR", "score": None, "reason": "API Failure", "judge_method": "none", "is_error": True}
            return {"verdict": "PASS", "score": 1.0, "reason": "Correct", "judge_method": "llm", "is_error": False}

        mock_judge.side_effect = mock_judge_impl

        report = run_eval(dataset_path=dataset_file, out_dir=tmp_path / "reports", skip_startup_check=True)

        # 20 queries, 1 error -> 19 valid queries
        assert report["total_queries"] == 20
        assert report["valid_queries"] == 19
        assert report["error_count"] == 1
        assert report["error_rate_pct"] == 5.0
        # Accuracy must be 100% of VALID queries (19/19), NOT 19/20 (95%)
        assert report["accuracy_pct"] == 100.0
        # Latency should be 1000ms (the error's 5000ms is excluded)
        assert report["latency_avg_ms"] == 1000.0
        assert report["is_valid"] is True


def test_eval_aborts_when_error_rate_exceeds_5_percent(tmp_path: Path):
    # Dataset with 10 items. 1 error is 10% > 5%, so it should abort
    dataset_file = tmp_path / "test_abort_dataset.jsonl"
    entries = []
    for i in range(10):
        entries.append({
            "id": f"Q{i+1:02d}",
            "question": f"Question {i+1}?",
            "ground_truth_answer": "Answer",
            "answerable": True,
            "category": "test"
        })
    with open(dataset_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    def mock_generate_rag(query: str, **kwargs):
        # Triggers infrastructure error
        return RAGResponse(
            answer="ERROR",
            citations=[],
            confidence=0.0,
            used_fallback=True,
            telemetry=TelemetryData(duration_s=1.0, ttft_ms=100.0, computed_cost_usd=0.0),
            answer_source="gemini",
            is_error=True,
            error_message="429 ResourceExhausted"
        )

    with patch("evals.run_evals.generate_rag_response", side_effect=mock_generate_rag), \
         patch("evals.run_evals.judge_answer") as mock_judge:
        mock_judge.return_value = {"verdict": "ERROR", "score": None, "reason": "API Failure", "judge_method": "none", "is_error": True}

        with pytest.raises(RuntimeError) as exc_info:
            run_eval(dataset_path=dataset_file, out_dir=tmp_path / "reports", skip_startup_check=True)

        assert "CRITICAL: Evaluation run aborted!" in str(exc_info.value)
        assert "exceeded 5.0% threshold" in str(exc_info.value)


def test_eval_marks_invalid_when_extractive_fallback_exceeds_5_percent(tmp_path: Path):
    # Dataset with 20 answerable items.
    # 2 extractive fallbacks = 10% > 5%, so run should be marked INVALID
    dataset_file = tmp_path / "test_invalid_dataset.jsonl"
    entries = []
    for i in range(20):
        entries.append({
            "id": f"Q{i+1:02d}",
            "question": f"Question {i+1}?",
            "ground_truth_answer": "Answer",
            "answerable": True,
            "category": "test"
        })
    with open(dataset_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    def mock_generate_rag(query: str, **kwargs):
        # 2 queries use extractive fallback
        if "Question 1?" in query or "Question 2?" in query:
            ans_src = "extractive_fallback"
        else:
            ans_src = "gemini"

        return RAGResponse(
            answer="Answer",
            citations=[],
            confidence=0.8,
            used_fallback=False,
            telemetry=TelemetryData(duration_s=0.5, ttft_ms=50.0, computed_cost_usd=0.0001),
            answer_source=ans_src,
            is_error=False
        )

    with patch("evals.run_evals.generate_rag_response", side_effect=mock_generate_rag), \
         patch("evals.run_evals.judge_answer") as mock_judge:
        mock_judge.return_value = {"verdict": "PASS", "score": 1.0, "reason": "Pass", "judge_method": "llm", "is_error": False}

        report = run_eval(dataset_path=dataset_file, out_dir=tmp_path / "reports", skip_startup_check=True)

        assert report["is_valid"] is False
        assert "INVALID" in report["validity_status"]
        assert report["extractive_fallback_on_answerable_count"] == 2
        assert report["extractive_fallback_on_answerable_pct"] == 10.0


def test_eval_resume_skips_successful_and_retries_errors(tmp_path: Path):
    # Dataset with 4 items
    dataset_file = tmp_path / "test_resume_dataset.jsonl"
    entries = [
        {"id": "Q01", "question": "Question 1?", "ground_truth_answer": "Answer 1", "answerable": True, "category": "test"},
        {"id": "Q02", "question": "Question 2?", "ground_truth_answer": "Answer 2", "answerable": True, "category": "test"},
        {"id": "Q03", "question": "Question 3?", "ground_truth_answer": "Answer 3", "answerable": True, "category": "test"},
        {"id": "Q04", "question": "Question 4?", "ground_truth_answer": "Answer 4", "answerable": True, "category": "test"}
    ]
    with open(dataset_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Initial checkpoint: Q01 succeeded, Q02 is an ERROR, Q03 and Q04 are missing
    chk_file = tmp_path / "checkpoint.json"
    initial_checkpoint = [
        {
            "id": "Q01",
            "category": "test",
            "question": "Question 1?",
            "answerable": True,
            "candidate_answer": "Answer 1",
            "ground_truth_answer": "Answer 1",
            "passed": True,
            "verdict": "PASS",
            "is_error": False,
            "error_message": None,
            "answer_source": "gemini",
            "judge_method": "llm",
            "judge_reason": "Correct",
            "model": settings.CHAT_MODEL,
            "harness_version": HARNESS_VERSION,
            "citations_precision": 1.0,
            "citations_recall": 1.0,
            "used_fallback": False,
            "duration_ms": 100.0,
            "ttft_ms": 50.0,
            "cost_usd": 0.0001
        },
        {
            "id": "Q02",
            "category": "test",
            "question": "Question 2?",
            "answerable": True,
            "candidate_answer": "ERROR",
            "ground_truth_answer": "Answer 2",
            "passed": None,
            "verdict": "ERROR",
            "is_error": True,
            "error_message": "429 ResourceExhausted",
            "answer_source": "gemini",
            "judge_method": "none",
            "judge_reason": "API Failure",
            "model": settings.CHAT_MODEL,
            "harness_version": HARNESS_VERSION,
            "citations_precision": None,
            "citations_recall": None,
            "used_fallback": True,
            "duration_ms": 0.0,
            "ttft_ms": 0.0,
            "cost_usd": 0.0
        }
    ]
    with open(chk_file, "w", encoding="utf-8") as f:
        json.dump(initial_checkpoint, f, indent=2)

    evaluated_queries = []

    def mock_generate_rag(query: str, **kwargs):
        evaluated_queries.append(query)
        return RAGResponse(
            answer="Generated Answer",
            citations=[],
            confidence=0.9,
            used_fallback=False,
            telemetry=TelemetryData(duration_s=0.2, ttft_ms=50.0, computed_cost_usd=0.0001),
            answer_source="gemini",
            is_error=False
        )

    with patch("evals.run_evals.generate_rag_response", side_effect=mock_generate_rag), \
         patch("evals.run_evals.judge_answer") as mock_judge:
        mock_judge.return_value = {"verdict": "PASS", "score": 1.0, "reason": "Correct", "judge_method": "llm", "is_error": False}

        report = run_eval(
            dataset_path=dataset_file,
            out_dir=tmp_path / "reports",
            resume=True,
            checkpoint_path=chk_file,
            skip_startup_check=True
        )

        # Q01 must NOT have been re-evaluated (it was reused from cache)
        assert "Question 1?" not in evaluated_queries
        # Q02 (the previous ERROR) MUST have been retried
        assert "Question 2?" in evaluated_queries
        # Q03 and Q04 (the missing ones) MUST have been evaluated
        assert "Question 3?" in evaluated_queries
        assert "Question 4?" in evaluated_queries

        # In the final report, all 4 queries should now be valid and passed!
        assert report["total_queries"] == 4
        assert report["valid_queries"] == 4
        assert report["error_count"] == 0
        assert report["accuracy_pct"] == 100.0


def test_is_valid_cache_entry_function():
    target_model = settings.CHAT_MODEL
    valid_entry = {
        "id": "Q01",
        "answer_source": "gemini",
        "judge_method": "llm",
        "model": target_model,
        "harness_version": HARNESS_VERSION,
        "candidate_answer": "Valid answer",
        "is_error": False,
        "verdict": "PASS"
    }

    # 1. Valid entry passes
    assert is_valid_cache_entry(valid_entry, target_model, HARNESS_VERSION) is True

    # 2. Missing answer_source rejected
    entry_no_ans_src = dict(valid_entry)
    del entry_no_ans_src["answer_source"]
    assert is_valid_cache_entry(entry_no_ans_src, target_model, HARNESS_VERSION) is False

    # 3. Invalid answer_source rejected
    entry_bad_ans_src = dict(valid_entry, answer_source="unknown_source")
    assert is_valid_cache_entry(entry_bad_ans_src, target_model, HARNESS_VERSION) is False

    # 4. Missing judge_method rejected
    entry_no_judge = dict(valid_entry)
    del entry_no_judge["judge_method"]
    assert is_valid_cache_entry(entry_no_judge, target_model, HARNESS_VERSION) is False

    # 5. Missing model rejected
    entry_no_model = dict(valid_entry)
    del entry_no_model["model"]
    assert is_valid_cache_entry(entry_no_model, target_model, HARNESS_VERSION) is False

    # 6. Mismatched model rejected
    entry_wrong_model = dict(valid_entry, model="different-model")
    assert is_valid_cache_entry(entry_wrong_model, target_model, HARNESS_VERSION) is False

    # 7. Missing harness_version rejected
    entry_no_ver = dict(valid_entry)
    del entry_no_ver["harness_version"]
    assert is_valid_cache_entry(entry_no_ver, target_model, HARNESS_VERSION) is False

    # 8. Older or different harness_version rejected
    entry_old_ver = dict(valid_entry, harness_version="2.0.0")
    assert is_valid_cache_entry(entry_old_ver, target_model, HARNESS_VERSION) is False

    # 9. Infrastructure error entry rejected
    entry_err = dict(valid_entry, is_error=True)
    assert is_valid_cache_entry(entry_err, target_model, HARNESS_VERSION) is False
    entry_sys_err = dict(valid_entry, candidate_answer="SYSTEM ERROR")
    assert is_valid_cache_entry(entry_sys_err, target_model, HARNESS_VERSION) is False


def test_cached_entry_without_answer_source_is_rejected(tmp_path: Path):
    """Proves that any cached entry lacking answer_source is rejected and re-run under --resume."""
    dataset_file = tmp_path / "test_no_ans_src_dataset.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "Q01",
            "question": "What is the engine spec?",
            "ground_truth_answer": "B58 3.0L inline-six engine",
            "answerable": True,
            "category": "specs"
        }) + "\n")

    # Cached checkpoint lacks 'answer_source'
    chk_file = tmp_path / "checkpoint_missing_ans_src.json"
    checkpoint_data = [
        {
            "id": "Q01",
            "category": "specs",
            "question": "What is the engine spec?",
            "answerable": True,
            "candidate_answer": "Stale cached answer",
            "ground_truth_answer": "B58 3.0L inline-six engine",
            "passed": True,
            "verdict": "PASS",
            "is_error": False,
            "error_message": None,
            # 'answer_source' is intentionally missing!
            "judge_method": "llm",
            "model": settings.CHAT_MODEL,
            "harness_version": HARNESS_VERSION,
            "citations_precision": 1.0,
            "citations_recall": 1.0,
            "used_fallback": False,
            "duration_ms": 100.0,
            "ttft_ms": 50.0,
            "cost_usd": 0.0001
        }
    ]
    with open(chk_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    evaluated_queries = []

    def mock_generate_rag(query: str, **kwargs):
        evaluated_queries.append(query)
        return RAGResponse(
            answer="B58 3.0L inline-six engine",
            citations=[],
            confidence=0.9,
            used_fallback=False,
            telemetry=TelemetryData(duration_s=0.2, ttft_ms=50.0, computed_cost_usd=0.0001),
            answer_source="gemini",
            is_error=False
        )

    with patch("evals.run_evals.generate_rag_response", side_effect=mock_generate_rag), \
         patch("evals.run_evals.judge_answer") as mock_judge:
        mock_judge.return_value = {"verdict": "PASS", "score": 1.0, "reason": "Correct", "judge_method": "llm", "is_error": False}

        report = run_eval(
            dataset_path=dataset_file,
            out_dir=tmp_path / "reports",
            resume=True,
            checkpoint_path=chk_file,
            skip_startup_check=True
        )

        # Because answer_source was missing, Q01 was REJECTED from cache and re-evaluated
        assert "What is the engine spec?" in evaluated_queries
        assert report["total_queries"] == 1
        assert report["valid_queries"] == 1
        assert report["detailed_results"][0]["candidate_answer"] == "B58 3.0L inline-six engine"
        assert report["detailed_results"][0]["answer_source"] == "gemini"


def test_startup_model_check():
    # 1. Missing api key raises RuntimeError
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY environment variable is missing"):
        startup_model_check("gemini-1.5-flash", "")

    # 2. 404 Not Found raises clear abort message
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_instance = MagicMock()
        mock_instance.generate_content.side_effect = Exception("404 models/gemini-old is not found")
        mock_model_cls.return_value = mock_instance

        with pytest.raises(RuntimeError) as exc_info:
            startup_model_check("gemini-old", "fake_key")
        assert "STARTUP CHECK FAILED: Model 'gemini-old' is not found (404)" in str(exc_info.value)

    # 3. 429 Quota Exhausted raises clear abort message
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_instance = MagicMock()
        mock_instance.generate_content.side_effect = Exception("429 Resource has been exhausted (check quota)")
        mock_model_cls.return_value = mock_instance

        with pytest.raises(RuntimeError) as exc_info:
            startup_model_check("gemini-1.5-flash", "fake_key")
        assert "STARTUP CHECK FAILED: API quota exceeded (429)" in str(exc_info.value)

    # 4. Successful call passes without exception
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MagicMock(text="pong")
        mock_model_cls.return_value = mock_instance

        # Should not raise
        startup_model_check("gemini-1.5-flash", "fake_key")
