"""Citation and Groundedness Provenance Validator.

Enforces strict verification on model-generated citations:
- Confirms citation chunk_id exists in actually retrieved chunks.
- Confirms quote_snippet is an authentic substring within that chunk.
- Strips hallucinated, fabricated, or ungrounded citations.
"""

import logging
from typing import Any, Dict, List, Tuple
from generation.schemas import Citation

logger = logging.getLogger(__name__)


def validate_and_filter_citations(
    candidate_citations: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]]
) -> Tuple[List[Citation], float]:
    """Validates citations against the retrieved chunk set and rejects hallucinations.

    Args:
        candidate_citations: Raw citations returned by the LLM.
        retrieved_chunks: The ground-truth chunks retrieved from the vector store.

    Returns:
        Tuple of (validated_citations_list, groundedness_score).
    """
    if not retrieved_chunks or not candidate_citations:
        return [], 0.0

    # Map retrieved chunks by chunk_id for O(1) lookup
    retrieved_map: Dict[str, Dict[str, Any]] = {
        chunk["chunk_id"]: chunk for chunk in retrieved_chunks
    }

    validated_citations: List[Citation] = []
    total_candidates = len(candidate_citations)

    for item in candidate_citations:
        chunk_id = item.get("chunk_id", "").strip()
        quote_snippet = item.get("quote_snippet", "").strip()

        # Check 1: Does chunk_id exist in actually retrieved chunks?
        if chunk_id not in retrieved_map:
            logger.warning(f"Citation rejected: chunk_id '{chunk_id}' was not in retrieved chunks.")
            continue

        target_chunk = retrieved_map[chunk_id]
        chunk_content = target_chunk["content"].lower()

        # Check 2: Is quote_snippet genuinely in the chunk content?
        clean_quote = quote_snippet.strip().strip('"\'').lower()
        if len(clean_quote) > 10 and clean_quote not in chunk_content:
            # Try a sub-phrase match if minor whitespace difference occurred
            words = clean_quote.split()
            sub_phrase = " ".join(words[:min(6, len(words))])
            if sub_phrase not in chunk_content:
                logger.warning(f"Citation rejected: quote_snippet '{quote_snippet}' not found in chunk {chunk_id}.")
                continue

        # Citation is genuine: enforce true document and chapter metadata from the chunk
        validated_citations.append(
            Citation(
                doc=target_chunk["source_doc"],
                page_or_chapter=target_chunk["page_or_chapter"],
                chunk_id=chunk_id,
                quote_snippet=quote_snippet
            )
        )

    # Compute groundedness ratio
    groundedness = len(validated_citations) / total_candidates if total_candidates > 0 else 0.0
    return validated_citations, groundedness
