"""
RAG Chat Engine using google-genai streaming.
AI service only performs retrieval and generation.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List, Literal, Optional

from google import genai

from app.core.config import settings
from app.services.rag.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

CreativityMode = Literal["strict", "expanded"]
DetailLevel = Literal["brief", "normal", "detailed"]

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("Google GenAI Client initialized")
    return _client


_BASE_PROMPT = """Bạn là trợ lý học tập cho hệ thống lớp học trực tuyến.

Mục tiêu:
- Trả lời tự nhiên như một trợ giảng đang giải thích cho sinh viên.
- Dùng tài liệu được cung cấp làm ngữ cảnh chính khi tài liệu có liên quan.
- Trả lời trực tiếp vào câu hỏi, không mở đầu bằng lời chào.
- Không nhắc đến "context", "chunk", "prompt" hoặc quá trình truy xuất.
- Không xuất thẻ <sources>, JSON nguồn, citation JSON hoặc metadata trong nội dung câu trả lời.
"""


def _answer_policy(creativity_mode: CreativityMode, detail_level: DetailLevel) -> str:
    if creativity_mode == "expanded":
        source_policy = """
Chế độ kiến thức: MỞ RỘNG.
- Được dùng kiến thức nền phổ biến ngoài tài liệu để giải thích khái niệm, so sánh, ví dụ như "ResNet-18 là gì" hoặc "ResNet-18 khác bản nâng cấp ở đâu".
- Nếu tài liệu có thông tin liên quan, hãy nêu phần trong tài liệu trước rồi mở rộng thêm.
- Nếu tài liệu không có đủ thông tin, vẫn trả lời bằng kiến thức nền phổ biến và nói ngắn gọn rằng tài liệu không nêu chi tiết phần đó.
- Tuyệt đối không trả lời bằng câu: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
- Không bịa chi tiết riêng của đồ án/lớp học/tài liệu nếu tài liệu không nói.
"""
    else:
        source_policy = """
Chế độ kiến thức: BÁM TÀI LIỆU.
- Chỉ trả lời dựa trên tài liệu được cung cấp.
- Nếu tài liệu chỉ có một phần thông tin, nói rõ phần chắc chắn từ tài liệu và phần không có dữ liệu.
- Nếu không tìm thấy thông tin trong tài liệu, trả lời đúng câu: "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
- Trả lời thân thiện, không cứng nhắc, nhưng không mở rộng thêm ngoài tài liệu.
"""

    detail_policy = {
        "brief": """
Mức độ chi tiết: NGẮN.
- Trả lời trong 1-2 đoạn ngắn hoặc tối đa 3 bullet.
- Ưu tiên định nghĩa và ý chính.
""",
        "normal": """
Mức độ chi tiết: VỪA.
- Trả lời đủ ý trong 2-4 đoạn ngắn.
- Có thể thêm bullet nếu câu hỏi hỏi danh sách hoặc so sánh.
""",
        "detailed": """
Mức độ chi tiết: CHI TIẾT.
- Giải thích theo tầng: định nghĩa, vai trò trong tài liệu, cách hoạt động/ý nghĩa, ví dụ ngắn nếu hữu ích.
- Dùng bullet rõ ràng khi có nhiều ý.
""",
    }[detail_level]

    return f"{source_policy}\n{detail_policy}"


def _build_context(chunks: List[dict]) -> str:
    if not chunks:
        return "Tài liệu được chọn không có đoạn nào đủ liên quan trực tiếp tới câu hỏi."

    parts = []
    for c in chunks:
        parts.append(f"[Trang {c['page_number']}]\n{c['chunk_text']}")
    return "\n\n---\n\n".join(parts)


async def stream_answer(
    user_id: int,
    class_id: int,
    question: str,
    document_ids: List[str],
    creativity_mode: CreativityMode = "strict",
    detail_level: DetailLevel = "normal",
) -> AsyncGenerator[str, None]:
    """
    Retrieve relevant chunks, ask Gemini, then emit SSE events.
    """
    chunks = hybrid_retrieve(question, document_ids, top_k=5)

    if not chunks and creativity_mode == "strict":
        no_doc_msg = (
            "Không tìm thấy nội dung liên quan trong tài liệu đã chọn. "
            "Hãy thử chọn thêm tài liệu khác hoặc đặt câu hỏi theo cách khác."
        )
        yield f"data: {no_doc_msg}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = _build_context(chunks)
    full_prompt = (
        f"{_BASE_PROMPT}\n\n"
        f"{_answer_policy(creativity_mode, detail_level)}\n\n"
        f"=== TÀI LIỆU ===\n{context}\n\n"
        f"=== CÂU HỎI ===\n{question}\n\n"
        "Hãy trả lời ngay."
    )

    try:
        client = _get_client()
        response = await client.aio.models.generate_content_stream(
            model="gemini-2.5-flash-lite",
            contents=full_prompt,
        )

        async for chunk in response:
            token = chunk.text or ""
            if token:
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

    except Exception as e:
        logger.error(f"GenAI SDK error: {e}")
        yield f"data: [ERROR] Lỗi khi gọi AI: {str(e)}\n\n"
        yield "data: [DONE]\n\n"
        return

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
