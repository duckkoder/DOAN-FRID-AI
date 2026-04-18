"""
RAG Chat Engine – Phase 4
Handles:
  1. History retrieval & question rephrasing (Gemini)
  2. Hybrid retrieval (Phase 3)
  3. Answer generation with SSE streaming (Gemini 1.5)
  4. Persisting messages to chat_sessions / chat_messages tables
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai

from app.core.config import settings
from app.services.rag.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client (lazy init)
# ---------------------------------------------------------------------------
_gemini_model: Optional[genai.GenerativeModel] = None


def _get_gemini() -> genai.GenerativeModel:
    global _gemini_model
    if _gemini_model is None:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini model initialized ✓")
    return _gemini_model


# ---------------------------------------------------------------------------
# DB helpers – chat session & message persistence
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(settings.DATABASE_URL)


def _get_or_create_session(user_id: int, class_id: int) -> str:
    """
    Return the existing chat session UUID for (user_id, class_id)
    or create a new one.  Works for both students and teachers.
    """
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Try to find existing session
            cur.execute(
                """
                SELECT id FROM chat_sessions
                WHERE student_id = %s AND course_id = %s::uuid
                LIMIT 1
                """,
                (user_id, str(class_id)),
            )
            row = cur.fetchone()
            if row:
                return str(row["id"])

            # Create new session
            session_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO chat_sessions (id, student_id, course_id)
                VALUES (%s, %s, %s::uuid)
                """,
                (session_id, user_id, str(class_id)),
            )
        conn.commit()
        logger.info(f"Created new chat session {session_id} for user {user_id}")
        return session_id
    finally:
        conn.close()


def _get_recent_history(session_id: str, n: int = 5) -> List[dict]:
    """Return the last n messages [{role, content}] for the session."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, n),
            )
            rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def _save_message(session_id: str, role: str, content: str) -> None:
    """Persist a message to the chat_messages table."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, content)
                VALUES (%s, %s, %s, %s)
                """,
                (str(uuid.uuid4()), session_id, role, content),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 1: Rephrase question using conversation history
# ---------------------------------------------------------------------------

async def _rephrase_question(history: List[dict], question: str) -> str:
    """
    Ask Gemini to turn the current question + history into a standalone query.
    Falls back to the original question on any error.
    """
    if not history:
        return question

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    prompt = (
        "Dưới đây là lịch sử cuộc hội thoại:\n"
        f"{history_text}\n\n"
        "Dựa vào lịch sử trên, hãy viết lại câu hỏi sau thành một câu hỏi "
        "hoàn chỉnh, độc lập, không cần ngữ cảnh trước đó. "
        "CHỈ trả về câu hỏi mới, KHÔNG giải thích thêm.\n\n"
        f"Câu hỏi gốc: {question}"
    )
    try:
        model = _get_gemini()
        response = await model.generate_content_async(prompt)
        rephrased = response.text.strip()
        logger.debug(f"Rephrased: {question!r} → {rephrased!r}")
        return rephrased
    except Exception as e:
        logger.warning(f"Rephrase failed ({e}), using original question")
        return question


# ---------------------------------------------------------------------------
# Step 2–3: Build prompt + stream answer
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là trợ lý giảng dạy thân thiện và chuyên nghiệp cho một hệ thống học tập trực tuyến.

NGUYÊN TẮC QUAN TRỌNG:
- Chỉ trả lời dựa trên nội dung TÀI LIỆU được cung cấp bên dưới.
- Nếu câu hỏi vượt ngoài phạm vi tài liệu, hãy nói: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
- TUYỆT ĐỐI KHÔNG bịa đặt hay suy diễn ngoài nội dung tài liệu.
- Trả lời bằng Tiếng Việt, rõ ràng, súc tích.
- Sau câu trả lời, hãy ghi rõ số trang nguồn theo định dạng JSON ở cuối.

Định dạng JSON sources (luôn đặt ở cuối, trong thẻ <sources>):
<sources>{"pages": [{"page": <số_trang>, "snippet": "<trích đoạn ngắn từ tài liệu>"}]}</sources>
"""


def _build_context(chunks: List[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[Trang {c['page_number']}]\n{c['chunk_text']}"
        )
    return "\n\n---\n\n".join(parts)


async def stream_answer(
    user_id: int,
    class_id: int,
    question: str,
    document_ids: List[str],
) -> AsyncGenerator[str, None]:
    """
    Main SSE generator.  Yields Server-Sent Event strings:
      - data: <token>\n\n          (text token)
      - data: [SOURCES] {...}\n\n  (final sources JSON)
      - data: [DONE]\n\n           (stream end signal)

    Args:
        user_id:      Authenticated user ID (student or teacher mapped to student_id col).
        class_id:     Class ID (integer, stored as UUID in course_id col).
        question:     Raw question text from the frontend.
        document_ids: UUIDs of selected documents.
    """
    # 1. Get / create session
    try:
        session_id = _get_or_create_session(user_id, class_id)
    except Exception as e:
        logger.error(f"Session error: {e}")
        yield f"data: [ERROR] Không thể khởi tạo phiên chat.\n\n"
        return

    # 2. Load history & rephrase
    history = _get_recent_history(session_id, n=5)
    rephrased = await _rephrase_question(history, question)

    # 3. Retrieve relevant chunks
    chunks = hybrid_retrieve(rephrased, document_ids, top_k=5)
    if not chunks:
        no_doc_msg = "Không tìm thấy nội dung liên quan trong tài liệu đã chọn. Hãy thử chọn thêm tài liệu khác hoặc đặt câu hỏi theo cách khác."
        _save_message(session_id, "user", question)
        _save_message(session_id, "ai", no_doc_msg)
        yield f"data: {no_doc_msg}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 4. Build prompt
    context = _build_context(chunks)
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"=== TÀI LIỆU ===\n{context}\n\n"
        f"=== CÂU HỎI ===\n{question}"
    )

    # 5. Stream from Gemini
    _save_message(session_id, "user", question)
    full_answer_parts: List[str] = []

    try:
        model = _get_gemini()
        response = await model.generate_content_async(
            full_prompt,
            stream=True,
        )

        async for chunk in response:
            token = chunk.text or ""
            if token:
                full_answer_parts.append(token)
                # SSE format
                # Escape newlines inside the SSE data field
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

    except Exception as e:
        logger.error(f"Gemini streaming error: {e}")
        yield f"data: [ERROR] Lỗi khi gọi AI: {str(e)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 6. Persist full answer
    full_answer = "".join(full_answer_parts)
    _save_message(session_id, "ai", full_answer)

    # 7. Emit sources as a structured JSON event
    sources_payload = {
        "pages": [
            {
                "page": c["page_number"],
                "document_id": str(c["document_id"]),
                "snippet": c["chunk_text"][:200],
            }
            for c in chunks
        ]
    }
    yield f"data: [SOURCES] {json.dumps(sources_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
