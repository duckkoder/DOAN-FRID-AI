import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # <-- Đã sửa (không dùng .notebook)
import warnings

# Tắt cảnh báo FutureWarning của torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# 1. CẤU HÌNH
# ===================================================================

# --- THÊM 'r' VÀO TRƯỚC ĐỂ SỬA LỖI ĐƯỜNG DẪN TRÊN WINDOWS ---
MODEL_WEIGHTS_PATH = r"D:\PBL6\FaceDetection\best_model_checkpoint.pth"
INPUT_FOLDER = r"D:\PBL6\FaceDetection\test"
OUTPUT_FOLDER = r"D:\PBL6\FaceDetection\res"

# Cấu hình model (PHẢI GIỐNG HỆT KHI TRAIN)
IMG_SIZE = 224
NUM_CLASSES = 2
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tên lớp (PHẢI ĐÚNG THỨ TỰ KHI TRAIN: {'real': 0, 'spoof': 1})
CLASS_NAMES = ['real', 'spoof']


# ===================================================================
# 2. ĐỊNH NGHĨA KIẾN TRÚC MODEL
# (Copy 100% từ notebook training)
# ===================================================================

# 2.1. BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x);
        out = self.bn1(out);
        out = self.relu(out)
        out = self.conv2(out);
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity;
        out = self.relu(out)
        return out


# 2.2. Kiến trúc Tối ưu (Sửa Stem + Multi-Scale Fusion)
class ResNet18_MSFF_AntiSpoof(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=NUM_CLASSES):
        super(ResNet18_MSFF_AntiSpoof, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256 * block.expansion + 512 * block.expansion, num_classes)  # 768
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [];
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x)
        x = self.layer1(x);
        x = self.layer2(x)
        f3 = self.layer3(x);
        f4 = self.layer4(f3)
        out3 = self.avgpool(f3);
        out4 = self.avgpool(f4)
        out3 = torch.flatten(out3, 1);
        out4 = torch.flatten(out4, 1)
        out = torch.cat((out3, out4), dim=1)
        out = self.fc(out)
        return out


# ===================================================================
# 3. HÀM CHẠY CHÍNH (MAIN FUNCTION)
# ===================================================================

def run_inference():
    print(f"Sử dụng thiết bị: {DEVICE}")
    print(f"Model: {MODEL_WEIGHTS_PATH}")
    print(f"Input: {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")

    # --- 3.1. TẢI MODEL ---
    print("\nĐang định nghĩa kiến trúc model...")
    print("   [OK] Đã định nghĩa BasicBlock và ResNet18_MSFF_AntiSpoof.")
    print("\nĐang tải model...")
    try:
        # Định nghĩa transform
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Khởi tạo model và tải trọng số
        model = ResNet18_MSFF_AntiSpoof(num_classes=NUM_CLASSES)

        # --- THÊM weights_only=True ĐỂ TẮT CẢNH BÁO ---
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH,
                                         map_location=DEVICE,
                                         weights_only=True))

        model.to(DEVICE)
        model.eval()  # BẮT BUỘC: Chuyển model sang chế độ đánh giá
        print("   [OK] Tải model và trọng số thành công!")

    except FileNotFoundError:
        print(f"\nLỖI NGHIÊM TRỌNG: Không tìm thấy file trọng số tại: {MODEL_WEIGHTS_PATH}")
        return
    except Exception as e:
        print(f"\nLỗi khi tải model: {e}")
        print("Hãy đảm bảo kiến trúc model (phần 2) khớp 100% với model đã lưu.")
        return

    # --- 3.2. CHẠY PHÂN LOẠI VÀ XUẤT KẾT QUẢ ---
    print(f"\nĐang tạo thư mục đầu ra tại: {OUTPUT_FOLDER}")
    # Xóa thư mục cũ (nếu có) và tạo thư mục mới
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(os.path.join(OUTPUT_FOLDER, CLASS_NAMES[0]), exist_ok=True)  # .../res/real
    os.makedirs(os.path.join(OUTPUT_FOLDER, CLASS_NAMES[1]), exist_ok=True)  # .../res/spoof

    # Thu thập tất cả ảnh
    image_files = []
    # Quét đệ quy (quét cả thư mục con)
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    print(f"Tìm thấy {len(image_files)} ảnh để phân loại.")
    if len(image_files) == 0:
        print(f"LỖI: Không tìm thấy ảnh nào trong {INPUT_FOLDER}. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Vòng lặp phân loại
    for img_path in tqdm(image_files, desc="Đang phân loại"):
        try:
            # Mở ảnh
            img = Image.open(img_path).convert('RGB')

            # Tiền xử lý (transform)
            input_tensor = test_transform(img)
            input_tensor = input_tensor.unsqueeze(0)  # Thêm batch dimension [1, 3, 224, 224]
            input_tensor = input_tensor.to(DEVICE)

            # Chạy dự đoán
            with torch.no_grad():  # Không cần tính gradient
                output = model(input_tensor)
                # Lấy class_id (0 hoặc 1)
                _, pred_idx = torch.max(output, 1)

            pred_class_name = CLASS_NAMES[pred_idx.item()]

            # Sao chép file vào thư mục kết quả
            output_filename = os.path.basename(img_path)
            target_path = os.path.join(OUTPUT_FOLDER, pred_class_name, output_filename)
            shutil.copy2(img_path, target_path)

        except Exception as e:
            print(f"\nLỗi khi xử lý file {img_path}: {e}")

    print("\n====================")
    print("PHÂN LOẠI HOÀN TẤT!")
    print(f"Kết quả đã được lưu tại: {OUTPUT_FOLDER}")
    print("====================")


# ===================================================================
# 4. GỌI HÀM MAIN
# ===================================================================
if __name__ == "__main__":
    run_inference()