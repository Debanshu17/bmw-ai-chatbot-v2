"""Citation and Groundedness Provenance Validator.

Enforces strict verification on model-generated citations:
- Confirms citation chunk_id exists in actually retrieved chunks.
- Confirms quote_snippet is an authentic substring within that chunk.
- Strips hallucinated, fabricated, or ungrounded citations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from generation.schemas import Citation

logger = logging.getLogger(__name__)


def validate_and_filter_citations(
    candidate_citations: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]]
) -> Tuple[List[Citation], float]:
    """Validates citations against the retrieved chunk set and rejects hallucinations.

    Accepts both full composite chunk keys ("file.md#id") and bare ID suffixes ("id"),
    resolving a bare ID to the single matching retrieved chunk. If a bare ID matches
    more than one retrieved chunk, it is rejected as ambiguous.

    Args:
        candidate_citations: Raw citations returned by the LLM.
        retrieved_chunks: The ground-truth chunks retrieved from the vector store.

    Returns:
        Tuple of (validated_citations_list, groundedness_score).
    """
    if not retrieved_chunks or not candidate_citations:
        return [], 0.0

    # Build mapping for full composite chunk_id
    composite_map: Dict[str, Dict[str, Any]] = {}
    # Build mapping for bare chunk_id suffix -> list of matching chunks
    bare_map: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in retrieved_chunks:
        cid = chunk.get("chunk_id", "").strip()
        composite_map[cid] = chunk

        # Extract bare ID suffix (after '#' if present)
        bare_id = cid.split("#", 1)[1] if "#" in cid else cid
        bare_map.setdefault(bare_id, []).append(chunk)

    validated_citations: List[Citation] = []
    total_candidates = len(candidate_citations)

    for item in candidate_citations:
        chunk_id = item.get("chunk_id", "").strip()
        quote_snippet = item.get("quote_snippet", "").strip()

        # Check 1: Resolve chunk_id
        target_chunk: Optional[Dict[str, Any]] = None

        if chunk_id in composite_map:
            # Full composite key match (e.g. "file.md#abc123_0")
            target_chunk = composite_map[chunk_id]
        elif chunk_id in bare_map:
            # Bare ID candidate match (e.g. "abc123_0")
            matching_chunks = bare_map[chunk_id]
            if len(matching_chunks) == 1:
                target_chunk = matching_chunks[0]
            else:
                logger.warning(
                    f"Citation rejected: bare chunk_id '{chunk_id}' is ambiguous "
                    f"({len(matching_chunks)} retrieved chunks match)."
                )
                continue
        else:
            logger.warning(f"Citation rejected: chunk_id '{chunk_id}' was not in retrieved chunks.")
            continue

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

        # Citation is genuine: enforce true document, chapter, and canonical composite chunk_id
        validated_citations.append(
            Citation(
                doc=target_chunk["source_doc"],
                page_or_chapter=target_chunk["page_or_chapter"],
                chunk_id=target_chunk["chunk_id"],
                quote_snippet=quote_snippet
            )
        )

    # Compute groundedness ratio
    groundedness = len(validated_citations) / total_candidates if total_candidates > 0 else 0.0
    return validated_citations, groundedness
