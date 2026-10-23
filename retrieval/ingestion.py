"""Document Ingestion and Provenance Chunking Module.

Parses raw documents (Markdown, Text) into structured chunks with strict metadata
including source_doc, page_or_chapter, chunk_id, character offsets, and content hash.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List
from config import settings


def compute_hash(text: str) -> str:
    """Computes a SHA256 hexadecimal hash for a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def parse_markdown_by_chapters(file_path: Path) -> List[Dict[str, Any]]:
    """Parses a markdown document into distinct sections/chapters based on headers.

    Args:
        file_path: Path to the markdown document.

    Returns:
        List of section dictionaries containing title, content, and char start/end offsets.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    doc_name = file_path.name
    # Regex to find markdown headers (e.g. ## Chapter 1: Title)
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(full_text))

    sections: List[Dict[str, Any]] = []

    if not matches:
        sections.append({
            "source_doc": doc_name,
            "page_or_chapter": "General",
            "start_offset": 0,
            "end_offset": len(full_text),
            "text": full_text.strip()
        })
        return sections

    for i, match in enumerate(matches):
        chapter_title = match.group(2).strip()
        start_offset = match.start()
        end_offset = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_text = full_text[start_offset:end_offset].strip()

        sections.append({
            "source_doc": doc_name,
            "page_or_chapter": chapter_title,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "text": section_text
        })

    return sections


def chunk_section(
    section: Dict[str, Any],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Chunks a section text while preserving character offsets and strict metadata.

    Args:
        section: Section dict with text, source_doc, page_or_chapter, start_offset.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk dictionaries with strict provenance metadata.
    """
    text = section["text"]
    section_start = section["start_offset"]
    chunks: List[Dict[str, Any]] = []

    if len(text) <= chunk_size:
        chunk_id = f"{section['source_doc']}#{compute_hash(text[:50])}_0"
        chunks.append({
            "chunk_id": chunk_id,
            "source_doc": section["source_doc"],
            "page_or_chapter": section["page_or_chapter"],
            "char_offsets": {
                "start": section_start,
                "end": section_start + len(text)
            },
            "content": text
        })
        return chunks

    start = 0
    idx = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # If possible, break on a sentence boundary
        if end < len(text):
            sentence_break = text.rfind(". ", start, end)
            if sentence_break != -1 and sentence_break > start + (chunk_size // 2):
                end = sentence_break + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{section['source_doc']}#{compute_hash(chunk_text[:40])}_{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "source_doc": section["source_doc"],
                "page_or_chapter": section["page_or_chapter"],
                "char_offsets": {
                    "start": section_start + start,
                    "end": section_start + end
                },
                "content": chunk_text
            })
            idx += 1

        if end >= len(text):
            break
        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def ingest_all_documents(docs_dir: Path = settings.RAW_DOCS_DIR) -> List[Dict[str, Any]]:
    """Scans and chunks all markdown and text files in the raw documents directory.

    Returns:
        List of all processed chunks across all documents.
    """
    all_chunks: List[Dict[str, Any]] = []
    supported_extensions = [".md", ".txt"]

    for file_path in docs_dir.glob("*"):
        if file_path.suffix.lower() in supported_extensions and file_path.is_file():
            sections = parse_markdown_by_chapters(file_path)
            for sec in sections:
                chunks = chunk_section(sec)
                all_chunks.extend(chunks)

    return all_chunks
