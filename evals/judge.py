"""LLM-as-a-Judge and Fuzzy Matching Evaluation Module.

Evaluates candidate answer accuracy against verified ground truth using a strict,
deterministic evaluation rubric at temperature=0 with fuzzy-match fallback.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
import google.generativeai as genai
from config import settings
from evals.retry import call_gemini_with_retry
from guardrails.gates import FALLBACK_MESSAGE

logger = logging.getLogger(__name__)

# Documented Evaluation Rubric Prompt
JUDGE_SYSTEM_PROMPT = """You are an impartial, rigorous evaluation judge grading RAG chatbot responses.
Compare the Candidate Answer against the Ground Truth Answer for the given Question.

EVALUATION RUBRIC:
1. FACTUAL ACCURACY: Does the Candidate Answer contain the essential facts, numbers, units, and technical specifications present in the Ground Truth?
2. NO CONTRADICTIONS: Does the Candidate Answer avoid asserting contradictory claims?
3. FALLBACK HANDLING: If the Ground Truth is the refusal message:
   "I couldn't find this in the provided documents, so I can't answer it reliably."
   Then the candidate answer MUST also be a refusal or state that the information is unavailable. Answering from outside knowledge is an immediate FAIL.
4. SCORING:
   - PASS: The candidate answer is factually consistent and accurate according to the ground truth.
   - FAIL: The candidate answer contains false statements, missing critical facts, hallucinations, or answered an unanswerable question.

OUTPUT FORMAT:
Return strictly valid JSON:
{"verdict": "PASS" | "FAIL", "score": 1.0 | 0.0, "reason": "concise explanation"}
"""


def fuzzy_match_verdict(candidate: str, ground_truth: str, answerable: bool) -> Tuple[bool, float, str]:
    """Fallback evaluation using deterministic string, substring, and token overlap.

    Args:
        candidate: Candidate generated answer.
        ground_truth: Ground truth verified answer.
        answerable: Whether the question is answerable.

    Returns:
        Tuple of (passed: bool, score: float, reason: str).
    """
    clean_cand = candidate.strip().lower()
    clean_gt = ground_truth.strip().lower()

    # Case 1: Unanswerable query
    if not answerable:
        refusal_keywords = [
            "couldn't find", "cannot answer", "not found",
            "not mentioned", "no information", "unavailable",
            FALLBACK_MESSAGE.lower()
        ]
        is_refusal = any(kw in clean_cand for kw in refusal_keywords)
        if is_refusal:
            return True, 1.0, "Correctly fell back on unanswerable query."
        return False, 0.0, "Hallucinated or answered an out-of-scope question."

    # Case 2: Exact match
    if clean_cand == clean_gt:
        return True, 1.0, "Exact match."

    # Case 3: Token Jaccard / Overlap
    cand_tokens = set(re.findall(r"\w+", clean_cand))
    gt_tokens = set(re.findall(r"\w+", clean_gt))

    if not gt_tokens:
        return False, 0.0, "Empty ground truth."

    intersection = cand_tokens.intersection(gt_tokens)
    overlap_ratio = len(intersection) / len(gt_tokens)

    # Check for crucial numeric/unit matches
    gt_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean_gt))
    cand_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean_cand))
    numbers_match = gt_numbers.issubset(cand_numbers) if gt_numbers else True

    if overlap_ratio >= 0.50 and numbers_match:
        return True, 1.0, f"Passed fuzzy overlap ({overlap_ratio:.2f}) with matching key figures."

    return False, 0.0, f"Failed overlap threshold (overlap={overlap_ratio:.2f}, numbers_match={numbers_match})."


def judge_answer(
    question: str,
    candidate_answer: str,
    ground_truth_answer: str,
    answerable: bool,
    candidate_is_error: bool = False
) -> Dict[str, Any]:
    """Judges candidate answer using LLM-as-a-judge with retry and deterministic fallback.

    Args:
        question: User query.
        candidate_answer: Model output.
        ground_truth_answer: Gold standard answer.
        answerable: Ground truth answerability boolean.
        candidate_is_error: True if the candidate answer was aborted due to infrastructure error.

    Returns:
        Dict with 'verdict' ('PASS'|'FAIL'|'ERROR'), 'score' (1.0|0.0|None), 'reason',
        'judge_method' ('llm'|'fuzzy'), and 'is_error' (bool).
    """
    # Infrastructure error fast-path: never score API failures as wrong answers
    if candidate_is_error or candidate_answer.strip() == "ERROR":
        return {
            "verdict": "ERROR",
            "score": None,
            "reason": "Infrastructure error occurred during answer generation.",
            "judge_method": "none",
            "is_error": True
        }

    # Fast path for fallback constant match
    if candidate_answer.strip() == FALLBACK_MESSAGE and ground_truth_answer.strip() == FALLBACK_MESSAGE:
        return {
            "verdict": "PASS",
            "score": 1.0,
            "reason": "Exact fallback message match.",
            "judge_method": "fuzzy",
            "is_error": False
        }

    # Attempt LLM-as-a-Judge with exponential backoff retry (up to 5x)
    if settings.GEMINI_API_KEY:
        prompt = (
            f"Question: {question}\n"
            f"Ground Truth Answer: {ground_truth_answer}\n"
            f"Candidate Answer: {candidate_answer}\n"
            f"Is Answerable: {answerable}\n\n"
            "Grade the candidate answer."
        )

        def _invoke_judge():
            model = genai.GenerativeModel(
                model_name=settings.CHAT_MODEL,
                system_instruction=JUDGE_SYSTEM_PROMPT,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            return model.generate_content(prompt)

        resp, is_api_error, err_detail = call_gemini_with_retry(_invoke_judge, max_retries=5)

        if is_api_error:
            # Judge API failed after 5 retries: mark as ERROR (not a failure of the model under test)
            logger.error(f"LLM-as-a-judge API call failed after 5 retries: {err_detail}")
            return {
                "verdict": "ERROR",
                "score": None,
                "reason": f"Judge API failure persisted after 5 retries: {err_detail}",
                "judge_method": "llm",
                "is_error": True
            }

        try:
            data = json.loads(resp.text)
            verdict = data.get("verdict", "FAIL").upper()
            score = 1.0 if verdict == "PASS" else 0.0
            reason = data.get("reason", "Graded by LLM-as-a-Judge.")
            return {
                "verdict": verdict,
                "score": score,
                "reason": reason,
                "judge_method": "llm",
                "is_error": False
            }
        except Exception as e:
            logger.warning(f"LLM-as-a-judge JSON parsing failed ({e}). Reverting to fuzzy match.")

    # Fallback to deterministic fuzzy matcher
    passed, score, reason = fuzzy_match_verdict(candidate_answer, ground_truth_answer, answerable)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "score": score,
        "reason": f"Fuzzy fallback: {reason}",
        "judge_method": "fuzzy",
        "is_error": False
    }
