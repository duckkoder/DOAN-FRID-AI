"""
RAG Ingestion Endpoint – Phase 2
Backend calls this after uploading a document to trigger the ingestion pipeline.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import boto3
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header
from pydantic import BaseModel

from app.core.config import settings
from app.services.rag.document_processor import process_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag")


# ---------------------------------------------------------------------------
# Auth: shared secret between Backend and AI Service
# ---------------------------------------------------------------------------

def _verify_callback_secret(x_callback_secret: str = Header(...)):
    if x_callback_secret != settings.BACKEND_CALLBACK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid callback secret")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    document_id: str   # UUID of the document row in `documents` table
    # Either provide a local pdf_path (dev) OR an s3_key (production)
    pdf_path: str | None = None
    s3_key: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_inserted: int
    status: str


# ---------------------------------------------------------------------------
# Background ingestion task (downloads from S3 if needed)
# ---------------------------------------------------------------------------

def _run_ingestion(document_id: str, pdf_path: str | None, s3_key: str | None):
    """Download from S3 (if s3_key given) then run the full pipeline."""
    try:
        if s3_key:
            # Download from S3 to a temp file
            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
            suffix = Path(s3_key).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
            try:
                s3.download_file(settings.S3_MODEL_BUCKET, s3_key, tmp_path)
                logger.info(f"Downloaded s3://{settings.S3_MODEL_BUCKET}/{s3_key} → {tmp_path}")
                result = process_document(document_id, tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        elif pdf_path:
            result = process_document(document_id, pdf_path)
        else:
            logger.error(f"No pdf_path or s3_key provided for document {document_id}")
            return
        logger.info(f"Ingestion done: {result}")
    except Exception as e:
        logger.error(f"Ingestion failed for {document_id}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger document ingestion (called by Backend)",
    dependencies=[Depends(_verify_callback_secret)],
)
async def ingest_document(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accepts a document ID + either pdf_path (local) or s3_key (S3 bucket).
    Queues the full ingestion pipeline as a background task.

    Request headers:
        X-Callback-Secret: <BACKEND_CALLBACK_SECRET>
    """
    if not body.pdf_path and not body.s3_key:
        raise HTTPException(status_code=422, detail="Provide either pdf_path or s3_key")

    if body.pdf_path:
        from pathlib import Path as _Path
        if not _Path(body.pdf_path).exists():
            raise HTTPException(status_code=422, detail=f"PDF not found at {body.pdf_path}")

    background_tasks.add_task(_run_ingestion, body.document_id, body.pdf_path, body.s3_key)

    logger.info(f"Queued ingestion for document {body.document_id}")
    return IngestResponse(
        document_id=body.document_id,
        chunks_inserted=0,      # actual count logged async
        status="processing",
    )


@router.post(
    "/ingest/sync",
    response_model=IngestResponse,
    summary="Synchronous ingestion (for testing / small files)",
    dependencies=[Depends(_verify_callback_secret)],
)
async def ingest_document_sync(body: IngestRequest):
    """Same as /ingest but waits for completion before returning."""
    if not body.pdf_path and not body.s3_key:
        raise HTTPException(status_code=422, detail="Provide either pdf_path or s3_key")

    # Run blocking code in threadpool to avoid blocking the event loop
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _run_ingestion, body.document_id, body.pdf_path, body.s3_key
    )
    return IngestResponse(
        document_id=body.document_id,
        chunks_inserted=-1,   # actual count inside _run_ingestion
        status="done",
    )

