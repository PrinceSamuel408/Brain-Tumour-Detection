"""
Phase 7: MRI Gatekeeper — Binary Pre-Classifier
=================================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script trains a lightweight binary classifier that answers one question:
    "Is this image actually a brain MRI scan?"

The gatekeeper runs BEFORE the main tumor classifier to reject selfies, cats,
screenshots, and any other non-MRI upload.  This prevents the ResNet50 tumor
model from producing nonsense predictions on out-of-distribution images.

Architecture
────────────
MobileNetV3-Small (pretrained on ImageNet) with a frozen backbone and a
2-class replacement head.  Chosen for sub-10ms inference on a MacBook CPU.

Dataset
───────
    mri_gatekeeper_dataset/
        is_mri/      ← symlinked or copied brain MRI scans from Training/
        not_mri/     ← CIFAR-10 images (random everyday objects)

Because CIFAR-10 images are 32×32 and MRIs are ~512×512, ALL images are
resized to 224×224 before entering the network.  This is the standard
ImageNet input size and works well for MobileNetV3.

Training
────────
5 epochs is plenty — this is a trivially easy visual distinction (grayscale
medical scans vs. colour photographs of trucks, birds, etc.).

Output
──────
    mri_gatekeeper.pth   — saved model weights (< 5 MB)
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from sklearn.model_selection import train_test_split

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "mri_gatekeeper_dataset")
TRAINING_DIR   = os.path.join(BASE_DIR, "Training")
SAVE_PATH      = os.path.join(BASE_DIR, "mri_gatekeeper.pth")

IMG_SIZE       = 224
BATCH_SIZE     = 32
NUM_EPOCHS     = 5
LEARNING_RATE  = 1e-3
NUM_WORKERS    = 2

# How many MRI images to copy into is_mri/ (sampled from Training/)
MRI_SAMPLE_SIZE = 2000

# How many CIFAR-10 images to save into not_mri/
CIFAR_SAMPLE_SIZE = 2000

GATEKEEPER_CLASSES = ["is_mri", "not_mri"]


# ──────────────────────────────────────────────
# 3. DEVICE
# ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# 4. DATASET PREPARATION
# ──────────────────────────────────────────────
def prepare_dataset():
    """
    Build the gatekeeper dataset from two sources:

        is_mri/  ← random sample of brain MRI scans from Training/
        not_mri/ ← CIFAR-10 images saved as PNGs

    Both sources are resized to 224×224 at training time via transforms,
    NOT during preparation — we keep the original files so the pipeline
    is transparent and auditable.
    """
    is_mri_dir  = os.path.join(DATASET_DIR, "is_mri")
    not_mri_dir = os.path.join(DATASET_DIR, "not_mri")

    # ── Populate is_mri/ from Training/ ──────────────────
    existing_mri = os.listdir(is_mri_dir)
    if len(existing_mri) < 100:
        print("Populating is_mri/ from Training/ ...")
        all_mri_paths = []
        for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
            class_dir = os.path.join(TRAINING_DIR, class_name)
            for fname in os.listdir(class_dir):
                all_mri_paths.append(os.path.join(class_dir, fname))

        sampled = random.sample(all_mri_paths, min(MRI_SAMPLE_SIZE, len(all_mri_paths)))
        for i, src in enumerate(sampled):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(is_mri_dir, f"mri_{i:04d}{ext}")
            shutil.copy2(src, dst)
        print(f"  Copied {len(sampled)} MRI scans into is_mri/")
    else:
        print(f"  is_mri/ already has {len(existing_mri)} images — skipping.")

    # ── Populate not_mri/ from CIFAR-10 ──────────────────
    existing_cifar = os.listdir(not_mri_dir)
    if len(existing_cifar) < 100:
        print("Populating not_mri/ from CIFAR-10 ...")
        from PIL import Image as PILImage

        # Download CIFAR-10 via torchvision (cached after first download)
        cifar = datasets.CIFAR10(
            root=os.path.join(BASE_DIR, ".cifar10_cache"),
            train=True,
            download=True,
        )

        indices = random.sample(range(len(cifar)), min(CIFAR_SAMPLE_SIZE, len(cifar)))
        for i, idx in enumerate(indices):
            img, _ = cifar[idx]  # PIL Image, 32×32
            save_path = os.path.join(not_mri_dir, f"cifar_{i:04d}.png")
            img.save(save_path)
        print(f"  Saved {len(indices)} CIFAR-10 images into not_mri/")
    else:
        print(f"  not_mri/ already has {len(existing_cifar)} images — skipping.")

    # Summary
    final_mri   = len(os.listdir(is_mri_dir))
    final_other = len(os.listdir(not_mri_dir))
    print(f"\nDataset ready: {final_mri} MRI + {final_other} non-MRI = {final_mri + final_other} total")


# ──────────────────────────────────────────────
# 5. TRANSFORMS
# ──────────────────────────────────────────────
# Both CIFAR (32×32) and MRI (variable) images get resized to 224×224.
# ImageNet normalization is required for MobileNetV3.
#
# IMPORTANT: We do NOT apply Grayscale here — keeping RGB is essential.
# MRI scans loaded as RGB have identical R=G=B channels (achromatic),
# while everyday photos have distinct colour channels.  This colour
# signature is the strongest feature for distinguishing the two classes.
# The Lambda below ensures any grayscale input is promoted to 3-channel
# RGB without destroying colour information in already-RGB images.
def _ensure_rgb(img):
    """Convert grayscale/RGBA to 3-channel RGB without stripping colour."""
    return img.convert("RGB")

train_transform = transforms.Compose([
    transforms.Lambda(_ensure_rgb),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Lambda(_ensure_rgb),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# 6. MODEL
# ──────────────────────────────────────────────
def build_gatekeeper(device):
    """
    MobileNetV3-Small with frozen backbone and a 2-class head.

    Architecture:
        MobileNetV3-Small backbone (FROZEN, 1.5M params)
        └─ classifier → Linear(576, 2)  (TRAINABLE, ~1.2K params)

    The original classifier is:
        Sequential(
            Linear(576, 1024), Hardswish, Dropout(0.2),
            Linear(1024, 1000)
        )
    We replace it with a single Linear(576, 2) — this task is so easy
    that a complex head would just overfit.
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    # Freeze the entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[-1].in_features  # 1024 from the penultimate layer
    # Keep the first layers of the classifier (Linear 576→1024, Hardswish, Dropout)
    # and only replace the final Linear
    model.classifier[-1] = nn.Linear(in_features, 2)

    # Unfreeze the new head
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(device)


# ──────────────────────────────────────────────
# 7. TRAINING LOOP
# ──────────────────────────────────────────────
def train_gatekeeper():
    device = get_device()
    print(f"Device: {device}\n")

    # ── Prepare dataset ──────────────────────────
    prepare_dataset()

    # ── Load full dataset with ImageFolder ───────
    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=train_transform)
    print(f"\nClasses: {full_dataset.classes}")
    print(f"Class-to-idx: {full_dataset.class_to_idx}")

    # ── Stratified train/val split (80/20) ───────
    all_targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=all_targets,
        random_state=SEED,
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset  = datasets.ImageFolder(root=DATASET_DIR, transform=eval_transform)
    val_subset   = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    # ── Model, loss, optimizer ───────────────────
    model = build_gatekeeper(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Count parameters
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,} | Trainable: {trainable:,} "
          f"({100*trainable/total:.2f}%)\n")

    # ── Training ─────────────────────────────────
    print("=" * 60)
    print("TRAINING GATEKEEPER")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # ── Train phase ──────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

        train_loss = running_loss / total_samples
        train_acc  = 100.0 * correct / total_samples

        # ── Validation phase ─────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc   = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.1f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {SAVE_PATH}")
    print("=" * 60)


# ──────────────────────────────────────────────
# 8. INFERENCE HELPER (used by Streamlit app)
# ──────────────────────────────────────────────
def load_gatekeeper(device):
    """Load the trained gatekeeper model for inference."""
    model = build_gatekeeper(device)
    state_dict = torch.load(SAVE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def is_brain_mri(image, model, device):
    """
    Run the gatekeeper on a PIL image.

    Parameters
    ----------
    image   : PIL.Image — the uploaded image (any size, RGB or grayscale)
    model   : the loaded MobileNetV3 gatekeeper model
    device  : torch.device

    Returns
    -------
    is_mri      : bool — True if the image looks like a brain MRI
    confidence  : float — confidence percentage (0–100)
    """
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.nn.functional.softmax(logits, dim=1).squeeze()

    # ImageFolder sorts alphabetically: is_mri=0, not_mri=1
    is_mri_prob = probs[0].item() * 100
    predicted   = logits.argmax(dim=1).item()

    return predicted == 0, is_mri_prob


# ──────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_gatekeeper()
    print("\nPhase 7 (MRI Gatekeeper) complete.")
