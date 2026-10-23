"""Deterministic Fallback Guardrails Module.

Provides code-level pre-LLM and post-LLM gating mechanisms:
- Gate 1 (Pre-LLM): Evaluates top retrieval score against SIMILARITY_THRESHOLD.
  Short-circuits without LLM invocation if context is absent or below threshold.
- Gate 2 (Post-LLM): Verifies citation presence and detects refusal triggers,
  replacing ungrounded responses with the fixed fallback constant.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from config import settings
from generation.schemas import RAGResponse

logger = logging.getLogger(__name__)

# Canonical fixed fallback message defined in settings
FALLBACK_MESSAGE: str = settings.FALLBACK_MESSAGE

# Common refusal substrings indicating failure to answer from context
REFUSAL_INDICATORS = [
    "i cannot answer",
    "i could not find",
    "not mentioned in the provided",
    "provided documents do not",
    "no information",
    "insufficient information",
    "not stated in the context",
    "i don't have access to"
]


def check_retrieval_gate(
    retrieved_chunks: List[Dict[str, Any]],
    threshold: float = settings.SIMILARITY_THRESHOLD
) -> Tuple[bool, Optional[str]]:
    """Gate 1 (Pre-LLM): Determines whether retrieved chunks meet minimum relevance.

    Args:
        retrieved_chunks: List of retrieved chunks with 'score' key.
        threshold: Minimum required similarity score for top chunk.

    Returns:
        Tuple of (passed_gate: bool, fallback_message: Optional[str]).
        If passed_gate is False, the LLM call MUST be skipped.
    """
    if not retrieved_chunks:
        logger.info("Gate 1 Triggered: No chunks retrieved. Returning fallback.")
        return False, FALLBACK_MESSAGE

    top_score = max(chunk.get("score", 0.0) for chunk in retrieved_chunks)
    if top_score < threshold:
        logger.info(
            f"Gate 1 Triggered: Top retrieval score {top_score:.4f} is below "
            f"threshold {threshold:.4f}. Short-circuiting LLM call."
        )
        return False, FALLBACK_MESSAGE

    return True, None


def check_generation_gate(response: RAGResponse) -> RAGResponse:
    """Gate 2 (Post-LLM): Validates generated response groundedness and citations.

    If citations are missing, or response indicates an inability to answer,
    the response answer is replaced with the fixed FALLBACK_MESSAGE.

    Args:
        response: Model-generated RAGResponse.

    Returns:
        Cleaned and gated RAGResponse.
    """
    # Check 1: Missing or stripped citations
    if not response.citations:
        logger.info("Gate 2 Triggered: Response contains 0 valid citations. Enforcing fallback.")
        response.answer = FALLBACK_MESSAGE
        response.used_fallback = True
        response.confidence = 0.0
        return response

    # Check 2: Explicit refusal phrases inside generated text
    lower_ans = response.answer.lower()
    for phrase in REFUSAL_INDICATORS:
        if phrase in lower_ans:
            logger.info(f"Gate 2 Triggered: Detected refusal phrase '{phrase}'. Enforcing fallback.")
            response.answer = FALLBACK_MESSAGE
            response.used_fallback = True
            response.citations = []
            response.confidence = 0.0
            return response

    return response


def get_system_prompt() -> str:
    """Returns the zero-knowledge-leakage system instruction prompt."""
    return (
        "You are the BMW M High-Performance Engineering Technical Assistant.\n"
        "Your task is to answer technical questions strictly and exclusively using the provided "
        "Context Chunks below.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using the facts stated in the Context Chunks. Do NOT use outside knowledge, "
        "assumptions, or external training data.\n"
        "2. If the context does not contain the answer, state clearly: "
        f"'{FALLBACK_MESSAGE}' and do not guess.\n"
        "3. For every claim made, you MUST cite the source chunk using its exact chunk_id and "
        "provide an exact verbatim quote_snippet from that chunk.\n"
        "4. Return your output formatted as valid JSON adhering to this schema:\n"
        "{\n"
        '  "answer": "Concise factual answer",\n'
        '  "citations": [{"doc": "filename", "page_or_chapter": "chapter name", "chunk_id": "exact_id", "quote_snippet": "exact quote"}],\n'
        '  "confidence": 0.95,\n'
        '  "used_fallback": false\n'
        "}"
    )
