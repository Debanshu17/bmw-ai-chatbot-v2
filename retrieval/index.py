"""Vector Store and Similarity Retrieval Index.

Provides persistent storage, vector indexing, cosine similarity scoring,
and top-k candidate chunk retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from config import settings
from retrieval.embeddings import get_embedding, get_query_embedding
from retrieval.ingestion import ingest_all_documents

logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory and file-persisted vector store utilizing NumPy cosine similarity."""

    def __init__(self, index_file: Path = settings.VECTOR_INDEX_FILE):
        self.index_file = index_file
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 3072), dtype=np.float32)

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Loads index from disk if present, otherwise ingests documents and builds it."""
        if not force_rebuild and self.index_file.exists():
            try:
                self.load()
                if len(self.chunks) > 0 and len(self.embeddings) == len(self.chunks):
                    logger.info(f"Loaded existing vector index with {len(self.chunks)} chunks from {self.index_file}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Rebuilding...")

        self.build()

    def build(self) -> None:
        """Ingests raw documents, generates embeddings, and saves index to disk."""
        logger.info("Ingesting raw documents...")
        self.chunks = ingest_all_documents()
        if not self.chunks:
            logger.warning("No raw documents found to ingest.")
            self.embeddings = np.empty((0, 768), dtype=np.float32)
            return

        logger.info(f"Embedding {len(self.chunks)} document chunks...")
        vectors = []
        for chunk in self.chunks:
            emb = get_embedding(chunk["content"])
            vectors.append(emb)

        self.embeddings = np.array(vectors, dtype=np.float32)
        # Normalize vectors for fast dot-product cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = self.embeddings / norms

        self.save()
        logger.info(f"Vector index successfully built and saved ({len(self.chunks)} chunks).")

    def save(self) -> None:
        """Persists chunks and embeddings to JSON."""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist()
        }
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self) -> None:
        """Loads chunks and embeddings from JSON."""
        with open(self.index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        self.embeddings = np.array(data.get("embeddings", []), dtype=np.float32)

    def search(self, query: str, top_k: int = settings.TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        """Searches index for top_k most similar chunks for a query string.

        Args:
            query: User search query.
            top_k: Number of candidate chunks to retrieve.

        Returns:
            List of chunk dicts annotated with similarity 'score'.
        """
        if len(self.chunks) == 0 or len(self.embeddings) == 0:
            return []

        q_vec = np.array(get_query_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity is the dot product of normalized vectors
        scores = np.dot(self.embeddings, q_vec)

        # Get top_k indices sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["score"] = float(scores[idx])
            results.append(chunk_copy)

        return results


# Global singleton instance
global_vector_index = VectorIndex()
