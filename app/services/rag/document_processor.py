"""
RAG Document Processor – Phase 2
Reads a PDF file → semantic chunking → GPU embedding → bulk insert to DB.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import psycopg2
from psycopg2.extras import execute_values
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton embedding model (loaded once at import time, stays on GPU)
# ---------------------------------------------------------------------------
_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load the vietnamese-bi-encoder model onto the configured device."""
    global _embed_model
    if _embed_model is None:
        device = settings.MODEL_DEVICE  # "cuda" or "cpu"
        model_name = settings.RAG_BI_ENCODER_MODEL
        logger.info(f"Loading {model_name} on device={device} …")
        _embed_model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model loaded ✓")
    return _embed_model


# ---------------------------------------------------------------------------
# PDF text extraction with page tracking
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract (page_number, text) tuples from a PDF.
    Uses PyMuPDF (fitz) for clean text and accurate page numbers.
    Returns a flat list of (page_no, text) – one entry per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []
    for page_index in range(len(doc)):
        text = doc[page_index].get_text("text").strip()
        if text:
            pages.append((page_index + 1, text))  # 1-indexed
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

def _semantic_chunk(
    pages: List[Tuple[int, str]],
    embed_model: SentenceTransformer,
    breakpoint_threshold_percentile: int = 95,
    max_chunk_chars: int = 2000,
) -> List[dict]:
    """
    Implement a lightweight SemanticChunker:
    - Splits each page into sentences.
    - Embeds every sentence.
    - Detects semantic breakpoints via cosine-distance spikes.
    - Groups sentences into chunks staying under max_chunk_chars.

    Returns list of dicts:
        { chunk_index, page_number, chunk_text, embedding }
    """
    import numpy as np

    def _split_sentences(text: str) -> List[str]:
        """Naive Vietnamese sentence splitter."""
        import re
        # Split on sentence-ending punctuation that is followed by a space or newline
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    # Flatten: list of (page_no, sentence)
    all_sentences: List[Tuple[int, str]] = []
    for page_no, text in pages:
        for sent in _split_sentences(text):
            all_sentences.append((page_no, sent))

    if not all_sentences:
        return []

    sentences_text = [s[1] for s in all_sentences]
    logger.info(f"Encoding {len(sentences_text)} sentences for semantic chunking …")

    # Encode all sentences in one batch for speed
    embeddings = embed_model.encode(
        sentences_text,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )  # shape (N, 768)

    # Consecutive cosine distance = 1 – dot product (since embeddings are normalized)
    dists = [
        1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Breakpoint = distance > threshold_percentile
    if dists:
        threshold = float(np.percentile(dists, breakpoint_threshold_percentile))
    else:
        threshold = 1.0

    # Build chunks
    chunks: List[dict] = []
    chunk_sentences: List[str] = []
    chunk_pages: List[int] = []
    chunk_embeddings_list = []

    def _flush_chunk(chunk_idx: int):
        if not chunk_sentences:
            return
        text_block = " ".join(chunk_sentences)
        # Average embedding of constituent sentences
        avg_emb = np.mean(chunk_embeddings_list, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-9)  # re-normalize
        chunks.append({
            "chunk_index": chunk_idx,
            "page_number": chunk_pages[0],   # page of first sentence
            "chunk_text": text_block,
            "embedding": avg_emb.tolist(),
        })

    chunk_idx = 0
    chunk_char_count = 0

    for i, (page_no, sent) in enumerate(all_sentences):
        chunk_sentences.append(sent)
        chunk_pages.append(page_no)
        chunk_embeddings_list.append(embeddings[i])
        chunk_char_count += len(sent)

        # Determine if we should break here
        is_last = i == len(all_sentences) - 1
        semantic_break = (i < len(dists)) and (dists[i] > threshold)
        size_break = chunk_char_count >= max_chunk_chars

        if is_last or semantic_break or size_break:
            _flush_chunk(chunk_idx)
            chunk_idx += 1
            chunk_sentences = []
            chunk_pages = []
            chunk_embeddings_list = []
            chunk_char_count = 0

    logger.info(f"Created {len(chunks)} semantic chunks ✓")
    return chunks


# ---------------------------------------------------------------------------
# Bulk insert into document_chunks
# ---------------------------------------------------------------------------

def _bulk_insert_chunks(document_id: str, chunks: List[dict]) -> int:
    """Insert chunks into the document_chunks table using psycopg2.
    Returns the number of rows inserted."""
    if not chunks:
        return 0

    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            rows = [
                (
                    str(uuid.uuid4()),
                    document_id,
                    c["chunk_index"],
                    c["page_number"],
                    c["chunk_text"],
                    c["embedding"],   # list[float] – psycopg2 + pgvector handles it
                )
                for c in chunks
            ]
            execute_values(
                cur,
                """
                INSERT INTO document_chunks
                    (id, document_id, chunk_index, page_number, chunk_text, embedding)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s::vector)",
            )
        conn.commit()
        logger.info(f"Inserted {len(rows)} chunks for document {document_id} ✓")
        return len(rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_document(document_id: str, pdf_path: str) -> dict:
    """
    Full ingestion pipeline for a single PDF document.

    Args:
        document_id: UUID string matching the `documents.id` in DB.
        pdf_path:    Absolute filesystem path to the PDF file.

    Returns:
        { "document_id": str, "chunks_inserted": int }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    embed_model = get_embed_model()
    pages = _extract_pages(str(path))
    logger.info(f"Extracted {len(pages)} pages from {path.name}")

    chunks = _semantic_chunk(pages, embed_model)
    inserted = _bulk_insert_chunks(document_id, chunks)

    return {"document_id": document_id, "chunks_inserted": inserted}
