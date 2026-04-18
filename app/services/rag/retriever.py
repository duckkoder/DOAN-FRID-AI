"""
RAG Hybrid Retriever – Phase 3
Combines:
  • Vector search  (pgvector HNSW, cosine similarity) — weight 60 %
  • Trigram search (pg_trgm GIN index)                — weight 40 %
Fuses scores with Reciprocal Rank Fusion (RRF) and returns top-K chunks.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from app.core.config import settings
from app.services.rag.document_processor import get_embed_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(settings.DATABASE_URL)


def _embed_query(query: str) -> List[float]:
    """Embed a single query string using the loaded vietnamese-bi-encoder."""
    model = get_embed_model()
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Vector retriever (HNSW cosine – pgvector)
# ---------------------------------------------------------------------------

def _vector_search(
    query_embedding: List[float],
    document_ids: List[str],
    top_k: int = 20,
) -> List[dict]:
    """
    Return top_k rows by cosine similarity.
    Mandatory filter on document_ids (the user's selected docs).
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.page_number,
                    dc.chunk_text,
                    1 - (dc.embedding <=> %s::vector) AS score
                FROM document_chunks dc
                WHERE dc.document_id = ANY(%s::uuid[])
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, document_ids, query_embedding, top_k),
            )
            rows = [dict(r) for r in cur.fetchall()]
        logger.debug(f"Vector search returned {len(rows)} results")
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Trigram retriever (pg_trgm)
# ---------------------------------------------------------------------------

def _trigram_search(
    query: str,
    document_ids: List[str],
    top_k: int = 20,
    sim_threshold: float = 0.05,
) -> List[dict]:
    """
    Return rows where trigram similarity > sim_threshold.
    Lower threshold (0.05) ensures we catch abbreviations / light typos.
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.page_number,
                    dc.chunk_text,
                    SIMILARITY(dc.chunk_text, %s) AS score
                FROM document_chunks dc
                WHERE dc.document_id = ANY(%s::uuid[])
                  AND SIMILARITY(dc.chunk_text, %s) > %s
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, document_ids, query, sim_threshold, top_k),
            )
            rows = [dict(r) for r in cur.fetchall()]
        logger.debug(f"Trigram search returned {len(rows)} results")
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def _rrf_fuse(
    vector_results: List[dict],
    trigram_results: List[dict],
    top_k: int = 5,
    k: int = 60,
    w_vector: float = 0.6,
    w_trigram: float = 0.4,
) -> List[dict]:
    """
    Fuse two ranked lists with Reciprocal Rank Fusion.
    Each chunk gets: score = w_vector / (k + rank_v) + w_trigram / (k + rank_t)
    Deduplicates by chunk id.
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, row in enumerate(vector_results, start=1):
        cid = str(row["id"])
        scores[cid] = scores.get(cid, 0.0) + w_vector / (k + rank)
        meta[cid] = row

    for rank, row in enumerate(trigram_results, start=1):
        cid = str(row["id"])
        scores[cid] = scores.get(cid, 0.0) + w_trigram / (k + rank)
        meta.setdefault(cid, row)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for cid, fused_score in ranked:
        chunk = meta[cid].copy()
        chunk["fused_score"] = fused_score
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    document_ids: List[str],
    top_k: int = 5,
) -> List[dict]:
    """
    Main retrieval function called by the chat engine.

    Args:
        query:        The (rephrased) user question.
        document_ids: List of document UUID strings the user selected.
        top_k:        Number of final chunks to return.

    Returns:
        List of chunk dicts sorted by relevance, each containing:
          id, document_id, chunk_index, page_number, chunk_text, fused_score
    """
    if not document_ids:
        logger.warning("hybrid_retrieve called with empty document_ids")
        return []

    query_embedding = _embed_query(query)
    vector_hits = _vector_search(query_embedding, document_ids, top_k=top_k * 4)
    trigram_hits = _trigram_search(query, document_ids, top_k=top_k * 4)
    fused = _rrf_fuse(vector_hits, trigram_hits, top_k=top_k)

    logger.info(
        f"hybrid_retrieve: query={query[:60]!r}, docs={len(document_ids)}, "
        f"vector={len(vector_hits)}, trigram={len(trigram_hits)}, fused={len(fused)}"
    )
    return fused
