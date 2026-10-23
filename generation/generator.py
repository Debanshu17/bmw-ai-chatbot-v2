"""End-to-End RAG Generation Pipeline.

Orchestrates retrieval, pre-LLM score gating, structured LLM synthesis,
in-code citation verification, post-LLM guardrails, and operational telemetry.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
import google.generativeai as genai
from config import settings
from generation.schemas import Citation, RAGResponse
from generation.validator import validate_and_filter_citations
from guardrails.gates import (
    FALLBACK_MESSAGE,
    check_generation_gate,
    check_retrieval_gate,
    get_system_prompt
)
from retrieval.index import global_vector_index
from telemetry.monitor import TelemetryTracker
from telemetry.storage import log_telemetry

logger = logging.getLogger(__name__)

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


def generate_rag_response(
    query: str,
    top_k: int = settings.TOP_K_CHUNKS,
    similarity_threshold: float = settings.SIMILARITY_THRESHOLD
) -> RAGResponse:
    """Executes the complete production RAG query lifecycle.

    Args:
        query: User technical question.
        top_k: Number of chunks to retrieve.
        similarity_threshold: Minimum required cosine similarity score.

    Returns:
        Structured, provenance-verified RAGResponse.
    """
    tracker = TelemetryTracker(model_name=settings.CHAT_MODEL).start()

    # Step 1: Retrieval
    retrieved_chunks = global_vector_index.search(query=query, top_k=top_k)

    # Step 2: Gate 1 (Pre-LLM Score Check)
    passed_gate1, fallback_msg = check_retrieval_gate(
        retrieved_chunks=retrieved_chunks,
        threshold=similarity_threshold
    )

    if not passed_gate1:
        # Short-circuit: 0 tokens, instant response
        telemetry = tracker.stop(prompt_tokens=0, completion_tokens=0)
        resp = RAGResponse(
            answer=fallback_msg or FALLBACK_MESSAGE,
            citations=[],
            confidence=0.0,
            used_fallback=True,
            telemetry=telemetry
        )
        log_telemetry(query=query, telemetry=telemetry, used_fallback=True, citations_count=0)
        return resp

    # Step 3: Format Context Chunks
    context_blocks = []
    for chunk in retrieved_chunks:
        block = (
            f"[CHUNK_ID: {chunk['chunk_id']}]\n"
            f"[SOURCE: {chunk['source_doc']} | {chunk['page_or_chapter']}]\n"
            f"{chunk['content']}\n"
        )
        context_blocks.append(block)

    full_context_str = "\n---\n".join(context_blocks)
    prompt = (
        f"Context Chunks:\n{full_context_str}\n\n"
        f"Question: {query}\n\n"
        "Generate a structured JSON response with your answer and citations."
    )

    # Step 4: Model Invocation
    raw_text = ""
    prompt_tokens = len(prompt.split()) * 2
    completion_tokens = 0

    try:
        model = genai.GenerativeModel(
            model_name=settings.CHAT_MODEL,
            system_instruction=get_system_prompt(),
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        tracker.record_first_token()
        response = model.generate_content(prompt)
        raw_text = response.text or ""
        completion_tokens = len(raw_text.split()) * 2

        # Parse JSON
        parsed_json = json.loads(raw_text)
        answer_text = parsed_json.get("answer", "")
        raw_citations = parsed_json.get("citations", [])

    except Exception as e:
        logger.warning(f"Gemini API call or JSON parsing failed ({e}). Extracting grounded answer from retrieved context.")
        # Graceful Extractive Fallback: Synthesize answer from top retrieved chunk
        top_chunk = retrieved_chunks[0]
        # Extract leading relevant sentences
        sentences = [s.strip() for s in top_chunk["content"].split(". ") if s.strip()]
        answer_text = ". ".join(sentences[:3]) + "."
        raw_citations = [
            {
                "chunk_id": top_chunk["chunk_id"],
                "doc": top_chunk["source_doc"],
                "page_or_chapter": top_chunk["page_or_chapter"],
                "quote_snippet": sentences[0] if sentences else top_chunk["content"][:80]
            }
        ]

    # Step 5: Citation Provenance Verification
    validated_citations, groundedness = validate_and_filter_citations(
        candidate_citations=raw_citations,
        retrieved_chunks=retrieved_chunks
    )

    # Step 6: Assemble Response
    top_score = max((c.get("score", 0.0) for c in retrieved_chunks), default=0.0)
    confidence = round(min(1.0, (groundedness * 0.6) + (top_score * 0.4)), 2)

    initial_response = RAGResponse(
        answer=answer_text,
        citations=validated_citations,
        confidence=confidence,
        used_fallback=False
    )

    # Step 7: Gate 2 (Post-LLM Grounding & Refusal Gate)
    final_response = check_generation_gate(initial_response)

    # Step 8: Telemetry finalization
    telemetry = tracker.stop(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    final_response.telemetry = telemetry

    log_telemetry(
        query=query,
        telemetry=telemetry,
        used_fallback=final_response.used_fallback,
        citations_count=len(final_response.citations)
    )

    return final_response
