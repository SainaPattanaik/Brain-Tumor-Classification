# src/predict.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F


# ------------------------------------------
#  Load Model
# ------------------------------------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]

    # Use new torchvision weights API for ResNet18
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, classes


# ------------------------------------------
#  Preprocessing (must match val transforms)
# ------------------------------------------
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------
#  Grad-CAM
# ------------------------------------------
def generate_gradcam(model, img_tensor, target_layer):
    img_tensor = img_tensor.unsqueeze(0)  # (1,3,224,224)

    # Forward hook
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    preds = model(img_tensor)
    class_idx = preds.argmax(dim=1).item()

    # Backprop
    model.zero_grad()
    preds[0, class_idx].backward()

    # Extract GRAD + ACT maps
    grads = gradients["value"]
    acts = activations["value"]

    weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP over HxW
    cam = (weights * acts).sum(dim=1)

    cam = F.relu(cam)
    cam = cam[0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = cv2.resize(cam, (224, 224))
    return cam, class_idx


# ------------------------------------------
#  Predict function
# ------------------------------------------
def predict_image(model, classes, img_path, device):
    img = Image.open(img_path).convert("RGB")
    inp = val_tf(img).to(device)

    with torch.no_grad():
        out = model(inp.unsqueeze(0))
        prob = torch.softmax(out, dim=1)[0]
        conf, pred_idx = torch.max(prob, 0)

    return classes[pred_idx.item()], conf.item(), inp


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "C:/Users/Harsha/Documents/GitHub/Brain-Tumor-Classification/models/best_model.pth"

    model, classes = load_model(model_path, device)

    img_path = "test.jpg"  # example
    pred, conf, inp = predict_image(model, classes, img_path, device)

    print("Prediction:", pred)
    print("Confidence:", conf)
