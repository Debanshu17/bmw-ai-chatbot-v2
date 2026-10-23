"""Unit tests for Pillar 4: Deterministic Fallback Guardrails."""

import pytest
from guardrails.gates import (
    FALLBACK_MESSAGE,
    check_retrieval_gate,
    check_generation_gate
)
from generation.schemas import Citation, RAGResponse


def test_retrieval_gate_empty_chunks():
    passed, message = check_retrieval_gate([], threshold=0.7)
    assert passed is False
    assert message == FALLBACK_MESSAGE


def test_retrieval_gate_below_threshold():
    chunks = [
        {"chunk_id": "doc1#0", "score": 0.45},
        {"chunk_id": "doc2#1", "score": 0.52}
    ]
    passed, message = check_retrieval_gate(chunks, threshold=0.68)
    assert passed is False
    assert message == FALLBACK_MESSAGE


def test_retrieval_gate_above_threshold():
    chunks = [
        {"chunk_id": "doc1#0", "score": 0.82},
        {"chunk_id": "doc2#1", "score": 0.65}
    ]
    passed, message = check_retrieval_gate(chunks, threshold=0.68)
    assert passed is True
    assert message is None


def test_generation_gate_zero_citations():
    resp = RAGResponse(
        answer="The S58 engine produces 510 horsepower.",
        citations=[],
        confidence=0.8,
        used_fallback=False
    )
    gated = check_generation_gate(resp)
    assert gated.used_fallback is True
    assert gated.answer == FALLBACK_MESSAGE
    assert gated.confidence == 0.0


def test_generation_gate_refusal_phrase_enforcement():
    resp = RAGResponse(
        answer="I cannot answer this question as the provided documents do not contain boost numbers.",
        citations=[
            Citation(
                doc="guide.md",
                page_or_chapter="Chapter 1",
                chunk_id="guide.md#0",
                quote_snippet="S58 engine specs"
            )
        ],
        confidence=0.5,
        used_fallback=False
    )
    gated = check_generation_gate(resp)
    assert gated.used_fallback is True
    assert gated.answer == FALLBACK_MESSAGE
    assert len(gated.citations) == 0


def test_generation_gate_valid_response_passes():
    resp = RAGResponse(
        answer="The S58 engine peak boost pressure is 2.1 bar (30.5 psi) in Competition models.",
        citations=[
            Citation(
                doc="bmw_m_s58_engine_technical_guide.md",
                page_or_chapter="Chapter 2",
                chunk_id="bmw_m_s58_engine_technical_guide.md#abc_0",
                quote_snippet="peak boost pressure of 2.1 bar"
            )
        ],
        confidence=0.95,
        used_fallback=False
    )
    gated = check_generation_gate(resp)
    assert gated.used_fallback is False
    assert gated.answer.startswith("The S58 engine peak boost")
    assert len(gated.citations) == 1
