"""Embedding endpoints.

AI service only computes vectors. Backend owns storage and tenant DB access.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.executor import get_model_executor

router = APIRouter(prefix="/embeddings")

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.RAG_BI_ENCODER_MODEL,
            device=settings.RAG_EMBEDDING_DEVICE,
        )
    return _model


class TextEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256)


class TextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int


@router.post("/text", response_model=TextEmbeddingResponse)
async def embed_texts(body: TextEmbeddingRequest):
    texts = [text.strip() for text in body.texts if text and text.strip()]
    if not texts:
        raise HTTPException(status_code=422, detail="texts cannot be empty")

    def _encode():
        model = _get_model()
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=settings.RAG_EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )

    vectors = await get_model_executor().execute_rag(_encode)
    embeddings = vectors.tolist()
    dimension = len(embeddings[0]) if embeddings else 0
    return TextEmbeddingResponse(embeddings=embeddings, dimension=dimension)
