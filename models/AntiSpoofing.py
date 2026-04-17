"""
Anti-Spoofing Model - ResNet18 với Multi-Scale Feature Fusion
Model mới train với 2 classes: real/spoof
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


# ============================================================
# BASIC BLOCK - ResNet Building Block
# ============================================================

class BasicBlock(nn.Module):
    """BasicBlock cho ResNet18"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 
            kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


# ============================================================
# RESNET18 MSFF (Multi-Scale Feature Fusion)
# ============================================================

class ResNet18_MSFF_AntiSpoof(nn.Module):
    """
    ResNet18 với Multi-Scale Feature Fusion cho Anti-Spoofing
    - Improved stem (3x3 conv thay vì 7x7)
    - Multi-scale fusion từ layer3 và layer4
    - Dropout regularization
    - 2 classes: real (0), spoof (1)
    """
    
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, dropout_rate=0.5):
        super(ResNet18_MSFF_AntiSpoof, self).__init__()
        
        self.in_planes = 64
        
        # ✅ Improved Stem - 3x3 conv thay vì 7x7
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # f3
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # f4
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ✅ Multi-Scale Fusion + Dropout + FC
        # 256 (layer3) + 512 (layer4) = 768 features
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * block.expansion + 512 * block.expansion, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Khởi tạo weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        
        # ✅ Multi-Scale Features
        f3 = self.layer3(x)   # 256 channels
        f4 = self.layer4(f3)  # 512 channels
        
        # Global pooling
        out3 = self.avgpool(f3)  # [B, 256, 1, 1]
        out4 = self.avgpool(f4)  # [B, 512, 1, 1]
        
        # Flatten
        out3 = torch.flatten(out3, 1)  # [B, 256]
        out4 = torch.flatten(out4, 1)  # [B, 512]
        
        # ✅ Concatenate multi-scale features
        out = torch.cat((out3, out4), dim=1)  # [B, 768]
        
        # FC + Dropout
        out = self.fc(out)  # [B, 2]
        
        return out


# ============================================================
# ANTI-SPOOFING CLASSIFIER - Inference Wrapper
# ============================================================

class AntiSpoofingClassifier:
    """
    Wrapper class cho Anti-Spoofing inference
    Model: ResNet18_MSFF_AntiSpoof
    Classes: real (0), spoof (1)
    """
    
    # ✅ Class mapping - PHẢI ĐÚNG THỨ TỰ TRAINING
    # Trong notebook, Pytorch ImageFolder sắp xếp folder theo bảng chữ cái:
    # 'fake' -> index 0, 'real' -> index 1
    CLASS_NAMES = ['spoof', 'real']  # 0: spoof (fake), 1: real
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Khởi tạo classifier
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: 'cuda' hoặc 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # ✅ Khởi tạo model với architecture mới
        self.model = ResNet18_MSFF_AntiSpoof(
            num_classes=2,
            dropout_rate=0.5
        ).to(self.device)
        
        # ✅ Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device, 
            weights_only=True
        )
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # ✅ Transform - PHẢI GIỐNG TRAINING CAO NHẤT CÓ THỂ
        # Lúc train dùng transforms.Resize((224, 224)), nếu dùng CenterCrop sẽ bị mất rìa ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Predict anti-spoofing label
        
        Args:
            image: PIL.Image, numpy.ndarray, hoặc file path
            
        Returns:
            tuple: (label, confidence)
                - label: 'real' hoặc 'spoof'
                - confidence: float (0.0 - 1.0)
        """
        # Convert to PIL Image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            # numpy array (BGR or RGB)
            if image.shape[2] == 3:
                img = Image.fromarray(image).convert('RGB')
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Preprocess
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        # Get label and confidence
        label = self.CLASS_NAMES[pred_idx.item()]
        confidence_score = confidence.item()
        
        return label, confidence_score