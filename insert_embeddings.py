import os
import sys
import psycopg2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# -----
# Thêm thư mục gốc vào sys.path để import app.services
# -----
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from app.services.face_recognition_service import FaceRecognitionService
    from app.services.embedding_manager import EmbeddingManager
    from app.core.config import settings
except ImportError:
    print("Lỗi: Không thể import 'app.services'.")
    print("Hãy đảm bảo bạn đặt script này trong thư mục gốc của project 'ai-service'")
    print(f"Thư mục gốc hiện tại: {project_root}")
    sys.exit(1)

# -------------------------------------------------
# --- CẤU HÌNH ---
# -------------------------------------------------

# 1. Đường dẫn đến folder chứa ảnh
IMAGE_FOLDER = r"E:\Workspace\PBL6\face-recognition\data_embedding_test_2"

# 2. Mapping student_code -> student_id (Sửa trực tiếp ở đây)
STUDENT_MAPPING = {
    "102220002": 19,
    "102220003": 18,

}

# 3. Đường dẫn đến model checkpoint
RECOGNIZER_CHECKPOINT = settings.RECOGNIZER_CHECKPOINT

# 4. Thông tin CSDL
DB_HOST = settings.POSTGRES_HOST
DB_PORT = settings.POSTGRES_PORT
DB_NAME = settings.POSTGRES_DB
DB_USER = settings.POSTGRES_USER
DB_PASS = settings.POSTGRES_PASSWORD

# 5. Tùy chọn tăng cường
USE_AUGMENTATION = True
AUGMENTATION_COUNT_PER_IMAGE = 5


# -------------------------------------------------
# --- FUNCTIONS ---
# -------------------------------------------------

def load_student_mapping(mapping_dict):
    """
    Chuyển đổi mapping dict sang format chuẩn

    Returns:
        dict: {student_code: student_id}
    """
    mapping = {}

    for code, sid in mapping_dict.items():
        mapping[str(code)] = int(sid)

    print(f"Đã tải {len(mapping)} mapping từ code.")
    return mapping


def get_db_connection():
    """Kết nối đến CSDL PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password="123qwe!@#"
        )
        return conn
    except Exception as e:
        print(f"Lỗi kết nối CSDL: {e}")
        sys.exit(1)


def insert_embedding(cursor, student_id, student_code, embedding_vector, image_path=None, status="approved"):
    """Chèn 1 vector embedding vào CSDL"""
    embedding_list = embedding_vector.tolist()

    # Xử lý giá trị mặc định cho image_path để tránh lỗi NOT NULL
    # Nếu không truyền vào path (ví dụ: ảnh augmented), gán là "augmented"
    final_image_path = image_path if image_path is not None else "test"

    insert_query = """
    INSERT INTO face_embeddings 
        (student_id, student_code, embedding, image_path, status, created_at, updated_at) 
    VALUES 
        (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    # Thêm final_image_path vào tuple tham số
    cursor.execute(insert_query, (student_id, student_code, str(embedding_list), final_image_path, status))


def main():
    print("=" * 60)
    print("SCRIPT CHÈN FACE EMBEDDINGS VÀO DATABASE")
    print("=" * 60)

    # 1. Load student mapping
    print("\n[1/4] Đang tải mapping student_code -> student_id...")
    student_mapping = load_student_mapping(STUDENT_MAPPING)

    # 2. Load model
    print("\n[2/4] Đang tải model Face Recognizer...")
    try:
        if not RECOGNIZER_CHECKPOINT:
            raise ValueError("RECOGNIZER_CHECKPOINT chưa được cấu hình.")

        os.environ.setdefault('RECOGNIZER_CHECKPOINT', RECOGNIZER_CHECKPOINT)
        recognizer = FaceRecognitionService()
        embedding_manager = EmbeddingManager()
        print("Tải model thành công.")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        sys.exit(1)

    # 3. Kết nối CSDL
    print(f"\n[3/4] Đang kết nối CSDL '{DB_NAME}' trên {DB_HOST}...")
    conn = get_db_connection()
    cursor = conn.cursor()
    print("Kết nối thành công.")

    # 4. Xử lý ảnh
    print(f"\n[4/4] Đang xử lý ảnh từ thư mục: {IMAGE_FOLDER}")
    source_dir = Path(IMAGE_FOLDER)
    if not source_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục ảnh: {IMAGE_FOLDER}")
        sys.exit(1)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    person_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    print(f"Tìm thấy {len(person_dirs)} thư mục sinh viên.\n")

    total_inserted = 0
    total_augmented = 0
    skipped_students = []

    for person_dir in tqdm(sorted(person_dirs), desc="Xử lý sinh viên"):
        student_code = person_dir.name

        # Lấy student_id từ mapping
        if student_code not in student_mapping:
            skipped_students.append(student_code)
            continue

        student_id = student_mapping[student_code]

        # Lấy danh sách ảnh
        image_files = [p for p in sorted(person_dir.iterdir())
                       if p.suffix.lower() in extensions]

        if not image_files:
            continue

        count_for_person = 0
        for image_path in image_files:
            try:
                # Trích xuất embedding từ ảnh gốc
                original_embedding = recognizer.extract_features(str(image_path), tta=True)
                insert_embedding(cursor, student_id, student_code, original_embedding)
                count_for_person += 1

                # Tạo ảnh tăng cường
                if USE_AUGMENTATION and AUGMENTATION_COUNT_PER_IMAGE > 0:
                    with Image.open(image_path) as img:
                        base_np = np.array(img.convert('RGB'))

                    aug_images = embedding_manager.generate_augmented_images(
                        base_np,
                        AUGMENTATION_COUNT_PER_IMAGE
                    )

                    for aug_img in aug_images:
                        aug_embedding = recognizer.extract_features(aug_img, tta=False)
                        insert_embedding(cursor, student_id, student_code, aug_embedding, image_path)
                        count_for_person += 1
                        total_augmented += 1

            except Exception as e:
                print(f"\nLỗi khi xử lý ảnh {image_path}: {e}")

        total_inserted += count_for_person

    # Commit và đóng kết nối
    conn.commit()
    cursor.close()
    conn.close()

    # In kết quả
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print("=" * 60)
    print(f"✓ Tổng số embedding đã chèn: {total_inserted}")
    print(f"  - Embedding gốc: {total_inserted - total_augmented}")
    print(f"  - Embedding tăng cường: {total_augmented}")
    print(f"✓ Số sinh viên đã xử lý: {len(person_dirs) - len(skipped_students)}")

    if skipped_students:
        print(f"\n⚠ Đã bỏ qua {len(skipped_students)} sinh viên không có trong mapping:")
        for code in skipped_students[:10]:  # Hiển thị tối đa 10
            print(f"  - {code}")
        if len(skipped_students) > 10:
            print(f"  ... và {len(skipped_students) - 10} sinh viên khác")

    print("=" * 60)


if __name__ == "__main__":
    main()