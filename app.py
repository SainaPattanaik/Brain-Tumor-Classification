# # app.py
# import streamlit as st
# import torch
# from PIL import Image
# import numpy as np
# import cv2
# import torchvision.models as models
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms

# from src.predict import load_model, predict_image, generate_gradcam


# st.set_page_config(
#     page_title="Brain Tumor MRI Classifier",
#     layout="centered",
#     page_icon="🧠"
# )

# st.title("🧠 Brain Tumor MRI Classification")
# st.write("Upload an MRI image and the model will classify it.")

# # Load model once
# @st.cache_resource
# def load():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, classes = load_model("C:/Users/Harsha/Documents/GitHub/Brain-Tumor-Classification/models/best_model.pth", device)
#     target_layer = model.layer4[1].conv2   # last conv layer for GradCAM
#     return model, classes, device, target_layer

# model, classes, device, target_layer = load()


# # Preprocessing (same as train/val)
# val_tf = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])


# uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# if uploaded:
#     st.image(uploaded, caption="Uploaded Image", use_column_width=True)

#     img = Image.open(uploaded).convert("RGB")
#     inp = val_tf(img).to(device)

#     # Prediction
#     with torch.no_grad():
#         out = model(inp.unsqueeze(0))
#         prob = torch.softmax(out, dim=1)[0]
#         conf, pred_idx = torch.max(prob, 0)

#     pred_label = classes[pred_idx.item()]
#     confidence = conf.item()

#     st.subheader("Prediction")
#     st.write(f"### **{pred_label}**")
#     st.write(f"Confidence: **{confidence:.3f}**")

#     # GradCAM
#     st.subheader("Grad-CAM Heatmap")

#     cam, class_idx = generate_gradcam(model, inp, target_layer)

#     cam = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

#     # Overlay heatmap
#     orig = np.array(img.resize((224, 224)))
#     overlay = (0.6 * cam + 0.4 * orig).astype(np.uint8)

#     st.image(overlay, caption="Grad-CAM", use_column_width=True)
# app.py

import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from src.predict import load_model, predict_image, generate_gradcam

# ----------------------------------------------------------
# Basic Page Config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    layout="centered",
    page_icon="🧠"
)

# ----------------------------------------------------------
# Custom CSS styling
# ----------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px !important;
        color: #4CAF50;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .pred-box {
        padding: 15px;
        border-radius: 10px;
        background: #f0f2f6;
        text-align: center;
        margin: 10px 0;
        border-left: 6px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload an MRI scan and let the model analyze it.</p>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Load model (cached)
# ----------------------------------------------------------
@st.cache_resource
def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(
        "C:/Users/Harsha/Documents/GitHub/Brain-Tumor-Classification/models/best_model.pth", 
        device
    )
    target_layer = model.layer4[1].conv2   # For Grad-CAM
    return model, classes, device, target_layer

model, classes, device, target_layer = load()

# Show device info
st.success(f"Model loaded successfully! Running on **{'GPU' if device=='cuda' else 'CPU'}**.")

# ----------------------------------------------------------
# Preprocessing (same as training)
# ----------------------------------------------------------
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ----------------------------------------------------------
# File Upload
# ----------------------------------------------------------
uploaded = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Processing indicator
    with st.spinner("Analyzing MRI scan..."):
        inp = val_tf(img).to(device)

        # Prediction
        with torch.no_grad():
            out = model(inp.unsqueeze(0))
            prob = torch.softmax(out, dim=1)[0]
            conf, pred_idx = torch.max(prob, 0)

        pred_label = classes[pred_idx.item()]
        confidence = conf.item()

        # GradCAM
        cam, _ = generate_gradcam(model, inp, target_layer)
        cam = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        orig = np.array(img.resize((224, 224)))
        overlay = (0.6 * cam + 0.4 * orig).astype(np.uint8)

    # ----------------------------------------------------------
    # Prediction Box
    # ----------------------------------------------------------
    st.markdown(f"""
    <div class='pred-box'>
        <h2>Prediction: <b style='color:#2E86C1'>{pred_label}</b></h2>
        <h4>Confidence: {confidence:.3f}</h4>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------
    # Show results in tabs
    # ----------------------------------------------------------
    tabs = st.tabs(["🔮 Prediction Details", "🔥 Grad-CAM Heatmap", "📊 Class Probabilities"])

    # Tab 1: Prediction
    with tabs[0]:
        st.write("### Model Output Probabilities:")
        result_dict = {cls: float(prob[i]) for i, cls in enumerate(classes)}
        st.json(result_dict)

    # Tab 2: Grad-CAM
    with tabs[1]:
        st.write("### Model Attention (Grad-CAM)")
        st.image(overlay, use_column_width=True)

    # Tab 3: Probability Bar Chart
    with tabs[2]:
        st.write("### Class Confidence Visualization")
        st.bar_chart(result_dict)

else:
    st.info("Please upload a brain MRI image to begin.")
