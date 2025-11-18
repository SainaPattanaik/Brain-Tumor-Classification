import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
# Assuming src.train defines build_model and DEVICE
# DEVICE should be defined as: torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import build_model, DEVICE 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# Removed the problematic import: from pytorch_grad_cam.utils.find_layers import find_target_layer 
import numpy as np

# Note: MODEL_PATH must exist in your file system
MODEL_PATH = "C:/Users/HP/OneDrive/Desktop/brain_tumor_project/models/best_model.pth"

# Use st.cache_resource for heavy components like models
@st.cache_resource
def load():
    """Loads the model and classes from the checkpoint."""
    try:
        # Load checkpoint to the correct device
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        classes = ckpt["classes"]
        # Build model architecture
        model = build_model(len(classes), feature_extract=True)
        # Load state dictionary
        model.load_state_dict(ckpt["model_state"])
        # Move model to the final, specified device (DEVICE) and set to evaluation mode
        # NOTE: Model starts in .eval() mode for inference, but will be briefly switched 
        # to .train() for Grad-CAM computation later.
        model.to(DEVICE).eval()
        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def preprocess(pil_img):
    """Applies necessary transformations to the PIL image for model input."""
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tf(pil_img).unsqueeze(0)


def to_rgb_float(pil_img):
    """Converts a PIL image to a normalized (0-1) NumPy array for Grad-CAM visualization."""
    # Resize to 224x224 for visualization consistency with the CAM
    arr = np.array(pil_img.resize((224,224))).astype(np.float32)/255.0
    # Handle grayscale images by stacking the channel
    if arr.ndim==2: arr = np.stack([arr]*3, axis=-1)
    return arr

st.title("Brain Tumor Classifier")
st.write("Upload a single MRI image (jpg/png).")

model, classes = load()

uploaded = st.file_uploader("Image", type=["jpg","jpeg","png"])
if uploaded:
    # 1. Image Loading and Preprocessing
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)
    
    # Move input tensor to the model's device (DEVICE)
    inp = preprocess(img).to(DEVICE)
    
    # CRITICAL FIX 1: Explicitly ensure the input tensor requires gradients. 
    # This often solves persistent 'NoneType' errors in Grad-CAM when hooks fail to activate.
    # We clone and detach just to isolate this operation cleanly from any upstream operations.
    inp = inp.detach().clone()
    inp.requires_grad_(True)
    
    # 2. Model Prediction
    # NOTE: The prediction still runs under no_grad() for efficient inference logging.
    # The cam() call handles the necessary gradient tracking later.
    with torch.no_grad():
        out = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        st.write(f"Prediction: *{classes[pred_idx]}* — confidence: {probs[pred_idx]:.3f}")

    # 3. Grad-CAM Visualization
    
    # FIX: Implement a robust search for the last convolutional layer.
    target_layer = None
    # Iterate over modules in reverse order to find the last Conv2d
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break

    if target_layer is None:
        st.error("Error: Could not find a suitable convolutional layer (Conv2d) in the model for Grad-CAM. This means the model structure is not recognizable by the automatic search.")
        st.stop()
        
    cam = GradCAM(model=model, target_layers=[target_layer]) 
    
    # Define the target for CAM (the predicted class index)
    targets = [ClassifierOutputTarget(pred_idx)]

    # --- Gradient Hook FIX 2 ---
    # Temporarily switch model to train() mode to ensure gradient hooks are active 
    # and collect the necessary gradients for the CAM computation.
    model.train()
    
    # Compute the CAM. Pass the target class index.
    grayscale_cam = cam(input_tensor=inp, targets=targets)[0]
    
    # Restore evaluation mode immediately after CAM calculation
    model.eval()
    # --- END FIX ---
    
    # Overlay the CAM on the original image
    cam_img = show_cam_on_image(to_rgb_float(img), grayscale_cam, use_rgb=True)
    st.image(cam_img, caption="Grad-CAM Visualization", use_column_width=True)