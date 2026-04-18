🚀 PHASE 1: Data Architecture & Infrastructure (Xây Móng)
Mục tiêu: Thiết lập cấu trúc lõi để lưu trữ dữ liệu đa chiều và chuẩn bị vũ khí tìm kiếm.

Chức năng 1: Thiết kế Database Schema bằng SQLAlchemy (bảng Course, Document, DocumentChunk, ChatSession, ChatMessage). Khóa chính full UUID.

Chức năng 2: Bật và cấu hình extension pgvector trên PostgreSQL, tạo Index HNSW cho cột embedding (768 chiều).

Chức năng 3: Bật extension pg_trgm, tạo Index GIN cho cột chunk_text để làm nền tảng chống sai chính tả.

🧠 PHASE 2: Data Ingestion (Nhồi Kiến Thức)
Mục tiêu: Xử lý file PDF thô thành các mảng tri thức có cấu trúc.

Chức năng 1: API Upload PDF, kiểm tra định dạng và lưu trữ vật lý.

Chức năng 2: Tích hợp SemanticChunker (LangChain) để băm tài liệu theo sự thay đổi ngữ nghĩa, trích xuất chính xác page_number và chunk_index.

Chức năng 3: Nhúng Vector bằng mô hình vietnamese-bi-encoder (chạy trên GPU) và Bulk Insert toàn bộ vào DB cùng một lúc để tối ưu I/O.

⚡ PHASE 3: Hybrid Search Engine (Não Bộ Truy Xuất)
Mục tiêu: Tìm kiếm bao chuẩn, bất chấp sinh viên gõ tắt, gõ ngọng hay sai chính tả.

Chức năng 1: Khởi tạo Vector Retriever tìm theo ngữ nghĩa (Trọng số 60%). Bắt buộc dùng Filter chặn theo course_id.

Chức năng 2: Viết custom SQL query dùng thuật toán Trigram Similarity (so khớp cụm 3 chữ cái) để bắt chuẩn Keyword và lỗi Typo (Trọng số 40%).

Chức năng 3: Trộn 2 luồng bằng EnsembleRetriever (LangChain) để lọc ra Top 5 chunk xuất sắc nhất, loại bỏ kết quả trùng lặp.

🤖 PHASE 4: LLM Generation & Streaming (Giao Tiếp Thời Gian Thực)
Mục tiêu: Tạo "linh hồn" cho Chatbot với trí nhớ và khả năng phản hồi mượt mà.

Chức năng 1: Xử lý Context: Lấy 5 tin nhắn gần nhất từ bảng ChatMessage, dùng Gemini viết lại thành câu hỏi độc lập (Rephrase).

Chức năng 2: Áp dụng System Prompt chứa cơ chế Fallback ("Không biết thì bảo không biết, cấm bịa chuyện").

Chức năng 3: Gọi API Gemini 1.5, trả kết quả về Frontend bằng kỹ thuật Server-Sent Events (SSE) để chữ hiện rào rào. Kèm theo cục JSON chứa sources (số trang).

🎨 PHASE 5: Frontend UI/UX (Mặt Tiền Tương Tác)
Mục tiêu: Biến các API khô khan thành trải nghiệm người dùng đẳng cấp.

Chức năng 1: Dựng Bảng tin (News Feed) cho lớp học, giảng viên có thể dặn dò và đính kèm tài liệu.

Chức năng 2: Dựng giao diện Split-pane: Cửa sổ Chatbot bên phải, Component @react-pdf-viewer bên trái.

Chức năng 3: Xử lý sự kiện OnClick: Khi sinh viên bấm vào "Nguồn: Trang X", tự động cuộn PDF tới trang đó và render khối bôi vàng (Highlight) đè lên mặt chữ.