# src/train.py

import os
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter


# ---------------- CONFIG ----------------
DATA_DIR = Path("C:/Users/HP/OneDrive/Desktop/brain_tumor_project/data")
EXPERIMENT_DIR = Path("../runs/exp1")
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
LR = 1e-4
EPOCHS = 12
# ----------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir, bs, workers):

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    test_path = data_dir / "test"
    test_ds = datasets.ImageFolder(test_path, transform=val_tf) if test_path.exists() else None

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=workers, pin_memory=True) if test_ds else None

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes, feature_extract=True):
    model = models.resnet18(pretrained=True)

    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()

            preds.extend(p.tolist())
            labels.extend(y.numpy().tolist())

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm
    }


def train():
    set_seed(SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(EXPERIMENT_DIR))

    train_loader, val_loader, test_loader, classes = make_loaders(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS
    )
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(num_classes, feature_extract=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_val = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        val_metrics = evaluate(model, val_loader, DEVICE)
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes
            }, EXPERIMENT_DIR / "best_model.pth")
            print("Saved best model.")

    # Test evaluation
    if test_loader:
        best = torch.load(EXPERIMENT_DIR / "best_model.pth", map_location=DEVICE)
        model.load_state_dict(best["model_state"])
        test_metrics = evaluate(model, test_loader, DEVICE)
        print("Test metrics:", test_metrics)
        writer.add_scalar("Accuracy/test", test_metrics["accuracy"], 0)

    writer.close()


if __name__ == "__main__":
    train()
