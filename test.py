import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from model import MITransformerModel
import os, random, re  # 🔍 Thêm re để xử lý regex

# 🎯 Hàm set_seed
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 ResNet50 feature extractor
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-2]
cnn_backbone = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1)))
feature_extractor = nn.Sequential(
    cnn_backbone,
    nn.Flatten(),
    nn.Linear(2048, 512)
).to(device)
feature_extractor.eval()

# 🧠 Load model MITransformer
model = MITransformerModel(img_dim=512, rad_dim=1, d_model=64, nhead=4, num_layers=2).to(device)
try:
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
except Exception as e:
    st.error(f"❌ Không thể load model: {e}")
    st.stop()

# 🎨 Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("📸 Dự báo bức xạ mặt trời")

# 🖼️ Tải chuỗi ảnh
uploaded_files = st.file_uploader(
    "Tải lên chuỗi ảnh và bức xạ đo được:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

radiance_values = []
if uploaded_files:
    st.subheader("📝 Nhập giá trị bức xạ quá khứ tương ứng với từng ảnh:")
    for file in uploaded_files:
        # ✅ Khớp số trước .jpg (ví dụ: 13.2 từ _13.2.jpg)
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\.jpg$', file.name.lower())
        try:
            default_val = float(match.group(1)) if match else 100.0
        except:
            default_val = 100.0

        val = st.number_input(
            f"Bức xạ quá khứ của {file.name} (W/m²):",
            min_value=0.0, value=default_val, step=0.1
        )
        radiance_values.append(val)
    st.subheader("🖼️ Ảnh đã tải lên:")
    cols = st.columns(min(5, len(uploaded_files)))
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        cols[i % 5].image(image, width=100, caption=f"Ảnh {i+1}")

    if st.button("📈 Dự đoán"):
        try:
            # 🧠 Trích xuất đặc trưng chuỗi ảnh
            img_feats = []
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = feature_extractor(img_tensor)  # (1, 512)
                    img_feats.append(feat)

            img_feat_seq = torch.stack(img_feats, dim=1)  # (1, T, 512)
            rad_seq = torch.tensor(radiance_values, dtype=torch.float32).view(1, -1, 1).to(device)

            # 🎯 Dự báo
            with torch.no_grad():
                output = model(img_feat_seq, rad_seq)
                prediction = output.item()

            st.subheader("🌞 Dự báo bức xạ mặt trời cho 5 phút sau:")
            st.success(f"🌤️ Giá trị bức xạ dự đoán: **{prediction:.2f} W/m²**")

        except Exception as e:
            st.error(f"❌ Lỗi dự đoán: {e}")
