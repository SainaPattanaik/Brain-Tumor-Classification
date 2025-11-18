# src/evaluate.py
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from src.train import build_model, DEVICE  # relative import, ensure PYTHONPATH or run from project root with python -m

DATA_DIR = Path("data")
MODEL_PATH = Path("models/best_model.pth")
BATCH_SIZE = 32

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    classes = ckpt["classes"]
    model = build_model(len(classes), feature_extract=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, classes

def run():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    model, classes = load_model(MODEL_PATH)
    preds, labels = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())

    print(classification_report(labels, preds, target_names=classes, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    run()