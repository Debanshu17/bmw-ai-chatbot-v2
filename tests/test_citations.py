"""Unit tests for Pillar 2: Citation Validation and Provenance Integrity."""

import pytest
from generation.validator import validate_and_filter_citations
from generation.schemas import Citation


@pytest.fixture
def sample_retrieved_chunks():
    return [
        {
            "chunk_id": "bmw_m_s58_engine_technical_guide.md#abc123_0",
            "source_doc": "bmw_m_s58_engine_technical_guide.md",
            "page_or_chapter": "Chapter 2: Twin-Turbocharging",
            "content": "Forced induction on the S58 engine delivers a maximum boost pressure of 2.1 bar (30.5 psi) in Competition models."
        },
        {
            "chunk_id": "bmw_m_chassis_xdrive_dynamics.md#def456_1",
            "source_doc": "bmw_m_chassis_xdrive_dynamics.md",
            "page_or_chapter": "Chapter 3: Braking Systems",
            "content": "The front carbon-ceramic discs measure 400 mm in diameter clamped by six-piston fixed calipers."
        }
    ]


def test_valid_citation_accepted(sample_retrieved_chunks):
    candidate = [
        {
            "chunk_id": "bmw_m_s58_engine_technical_guide.md#abc123_0",
            "quote_snippet": "maximum boost pressure of 2.1 bar (30.5 psi)"
        }
    ]
    validated, score = validate_and_filter_citations(candidate, sample_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].doc == "bmw_m_s58_engine_technical_guide.md"
    assert validated[0].page_or_chapter == "Chapter 2: Twin-Turbocharging"
    assert score == 1.0


def test_bare_id_resolves(sample_retrieved_chunks):
    """Bare ID suffix without filename prefix must resolve to the matching chunk."""
    candidate = [
        {
            "chunk_id": "abc123_0",
            "quote_snippet": "maximum boost pressure of 2.1 bar (30.5 psi)"
        }
    ]
    validated, score = validate_and_filter_citations(candidate, sample_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].doc == "bmw_m_s58_engine_technical_guide.md"
    assert validated[0].page_or_chapter == "Chapter 2: Twin-Turbocharging"
    assert validated[0].chunk_id == "bmw_m_s58_engine_technical_guide.md#abc123_0"
    assert score == 1.0


def test_composite_id_resolves(sample_retrieved_chunks):
    """Full composite chunk key ('file.md#id') must resolve directly."""
    candidate = [
        {
            "chunk_id": "bmw_m_chassis_xdrive_dynamics.md#def456_1",
            "quote_snippet": "front carbon-ceramic discs measure 400 mm"
        }
    ]
    validated, score = validate_and_filter_citations(candidate, sample_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].doc == "bmw_m_chassis_xdrive_dynamics.md"
    assert validated[0].chunk_id == "bmw_m_chassis_xdrive_dynamics.md#def456_1"
    assert score == 1.0


def test_unknown_id_stripped(sample_retrieved_chunks):
    """Unknown chunk ID must be stripped even if quote is authentic."""
    candidate = [
        {
            "chunk_id": "unknown_id_9999",
            "quote_snippet": "maximum boost pressure of 2.1 bar (30.5 psi)"
        }
    ]
    validated, score = validate_and_filter_citations(candidate, sample_retrieved_chunks)
    assert len(validated) == 0
    assert score == 0.0


def test_ambiguous_bare_id_is_rejected():
    """If a bare ID matches more than one retrieved chunk, it must be rejected as ambiguous."""
    retrieved = [
        {
            "chunk_id": "manual_part1.md#shared_chunk_0",
            "source_doc": "manual_part1.md",
            "page_or_chapter": "Section A",
            "content": "Engine oil cooling uses an auxiliary heat exchanger."
        },
        {
            "chunk_id": "manual_part2.md#shared_chunk_0",
            "source_doc": "manual_part2.md",
            "page_or_chapter": "Section B",
            "content": "Transmission oil cooling uses an auxiliary heat exchanger."
        }
    ]
    # Ambiguous bare ID
    candidate_bare = [
        {
            "chunk_id": "shared_chunk_0",
            "quote_snippet": "auxiliary heat exchanger"
        }
    ]
    validated, score = validate_and_filter_citations(candidate_bare, retrieved)
    assert len(validated) == 0
    assert score == 0.0

    # Composite ID disambiguates and must resolve
    candidate_composite = [
        {
            "chunk_id": "manual_part1.md#shared_chunk_0",
            "quote_snippet": "auxiliary heat exchanger"
        }
    ]
    validated_comp, score_comp = validate_and_filter_citations(candidate_composite, retrieved)
    assert len(validated_comp) == 1
    assert validated_comp[0].doc == "manual_part1.md"
    assert score_comp == 1.0


def test_fabricated_id_never_validates(sample_retrieved_chunks):
    """Fabricated chunk IDs (composite or bare) must never validate."""
    candidates = [
        {
            "chunk_id": "fabricated_bare_id_999",
            "quote_snippet": "maximum boost pressure of 2.1 bar"
        },
        {
            "chunk_id": "fake_doc.md#fabricated_chunk_888",
            "quote_snippet": "maximum boost pressure of 2.1 bar"
        }
    ]
    validated, score = validate_and_filter_citations(candidates, sample_retrieved_chunks)
    assert len(validated) == 0
    assert score == 0.0


def test_invented_chunk_id_stripped(sample_retrieved_chunks):
    candidate = [
        {
            "chunk_id": "invented_fake_doc.pdf#9999_0",
            "quote_snippet": "maximum boost pressure of 2.1 bar"
        }
    ]
    validated, score = validate_and_filter_citations(candidate, sample_retrieved_chunks)
    assert len(validated) == 0
    assert score == 0.0


def test_fabricated_quote_snippet_stripped(sample_retrieved_chunks):
    candidate = [
        {
            "chunk_id": "bmw_m_s58_engine_technical_guide.md#abc123_0",
            "quote_snippet": "The engine was developed by Mercedes AMG in Stuttgart"  # Fabricated snippet
        }
    ]
    validated, score = validate_and_filter_citations(candidate, sample_retrieved_chunks)
    assert len(validated) == 0
    assert score == 0.0


def test_mixed_candidates_only_valid_kept(sample_retrieved_chunks):
    candidates = [
        {
            "chunk_id": "bmw_m_s58_engine_technical_guide.md#abc123_0",
            "quote_snippet": "maximum boost pressure of 2.1 bar"
        },
        {
            "chunk_id": "hallucinated_chunk#000",
            "quote_snippet": "Random made-up text"
        }
    ]
    validated, score = validate_and_filter_citations(candidates, sample_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].chunk_id == "bmw_m_s58_engine_technical_guide.md#abc123_0"
    assert score == 0.5
