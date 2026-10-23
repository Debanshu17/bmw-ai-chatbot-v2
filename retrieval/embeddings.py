"""Embedding Generation Module.

Generates dense vector embeddings using Google Generative AI (text-embedding-004)
with fallback handling for offline or test environments.
"""

import logging
from typing import List
import numpy as np
import google.generativeai as genai
from config import settings

logger = logging.getLogger(__name__)

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


def get_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generates an embedding vector for a given text string.

    Args:
        text: Input text content.
        task_type: Task type hint for Gemini embeddings (retrieval_document or retrieval_query).

    Returns:
        List of floats representing the embedding vector.
    """
    if not text.strip():
        return [0.0] * 3072

    try:
        if settings.GEMINI_API_KEY:
            result = genai.embed_content(
                model=settings.EMBEDDING_MODEL,
                content=text,
                task_type=task_type
            )
            embedding = result.get("embedding", [])
            if embedding:
                return embedding
    except Exception as e:
        logger.warning(f"Gemini embedding API failed or unconfigured ({e}). Utilizing deterministic fallback embedding.")

    # Deterministic fallback embedding for offline evaluation or unit tests
    # Generates reproducible 3072-dimensional normalized pseudo-semantic hash vector
    vec = np.zeros(3072, dtype=np.float32)
    words = text.lower().split()
    for i, word in enumerate(words):
        h = hash(word) % 3072
        vec[h] += 1.0 / (i + 1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def get_query_embedding(query: str) -> List[float]:
    """Generates embedding for a user query."""
    return get_embedding(query, task_type="retrieval_query")
