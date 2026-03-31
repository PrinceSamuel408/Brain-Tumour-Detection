"""
Phase 1: Environment Setup and Data Pipeline
=============================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script builds a complete data loading and preprocessing pipeline for a
brain MRI classification task with four classes:
    - glioma
    - meningioma
    - notumor (healthy)
    - pituitary

Key design decisions:
    1. Images are resized to 224×224 to match standard pretrained model inputs.
    2. Grayscale MRIs are converted to 3-channel RGB so they work with models
       pretrained on ImageNet (which expects 3 channels).
    3. The original Training folder is re-split into 70 / 15 / 15 subsets using
       stratified splitting to preserve class balance.
    4. The original Testing folder is kept as a held-out evaluation set.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os
import copy
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Reproducibility — fix all random seeds so every run is identical.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────
# Paths — update these if your folder structure is different.
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR      = os.path.join(BASE_DIR, "Training")
TEST_DIR       = os.path.join(BASE_DIR, "Testing")

# Hyperparameters
IMG_SIZE       = 224          # target spatial resolution (224×224)
BATCH_SIZE     = 32
NUM_WORKERS    = 2            # parallel data-loading threads
TRAIN_RATIO    = 0.70         # 70 % training
VAL_RATIO      = 0.15         # 15 % validation  (remaining 15 % = test split)

# Human-readable class names (alphabetical — matches ImageFolder ordering)
CLASS_NAMES    = ["glioma", "meningioma", "notumor", "pituitary"]

# ──────────────────────────────────────────────
# 3. TRANSFORMS (PREPROCESSING PIPELINES)
# ──────────────────────────────────────────────
# Training transform — includes data augmentation to improve generalisation.
# Medical imaging benefits from geometric augmentations (flips, rotations)
# but NOT from colour jitter (colour carries diagnostic meaning in MRI scans).
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),       # standardise spatial size
    transforms.Grayscale(num_output_channels=3),   # ensure 3 channels (handles mixed grayscale/RGB)
    transforms.RandomHorizontalFlip(p=0.5),        # 50 % chance horizontal flip
    transforms.RandomRotation(degrees=15),         # slight rotation ±15°
    transforms.ToTensor(),                         # [H,W,C] uint8 → [C,H,W] float32 ∈ [0,1]
    transforms.Normalize(                          # ImageNet statistics — required by pretrained backbones
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation / Test transform — deterministic (no augmentation).
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ──────────────────────────────────────────────
# 4. DATASET LOADING & STRATIFIED SPLITTING
# ──────────────────────────────────────────────
# Load the full training directory with ImageFolder.
# ImageFolder automatically assigns integer labels based on sorted subfolder names.
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

# Extract all labels so we can do a stratified split (preserves class proportions).
all_targets = np.array(full_dataset.targets)
all_indices = np.arange(len(full_dataset))

# First split: separate out the training set (70 %) from the remaining 30 %.
train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=(1 - TRAIN_RATIO),          # 30 % goes to temp
    stratify=all_targets,                  # preserve class balance
    random_state=SEED,
)

# Second split: divide the remaining 30 % equally into validation (15 %) and
# test-from-train (15 %).  0.5 × 30 % = 15 %.
temp_targets = all_targets[temp_indices]
val_indices, test_split_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    stratify=temp_targets,
    random_state=SEED,
)

# Build Subset objects.  Validation and test subsets need the eval transform
# (no augmentation), so we create a second ImageFolder with eval_transform and
# take subsets from it.
eval_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=eval_transform)

train_subset = Subset(full_dataset,  train_indices)
val_subset   = Subset(eval_dataset,  val_indices)
test_subset  = Subset(eval_dataset,  test_split_indices)

# The original Testing folder serves as a completely held-out evaluation set.
holdout_test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transform)

# ──────────────────────────────────────────────
# 5. DATALOADERS
# ──────────────────────────────────────────────
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,              # shuffle every epoch for training
    num_workers=NUM_WORKERS,
    pin_memory=True,           # speed up host → GPU transfers
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

holdout_loader = DataLoader(
    holdout_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# ──────────────────────────────────────────────
# 6. SPLIT SUMMARY (printed to console)
# ──────────────────────────────────────────────
def print_split_summary():
    """Print the size and per-class distribution of each data split."""
    splits = {
        "Train (70 %)":        (train_indices, all_targets),
        "Validation (15 %)":   (val_indices, all_targets),
        "Test-split (15 %)":   (test_split_indices, all_targets),
        "Holdout-Test":        (
            np.arange(len(holdout_test_dataset)),
            np.array(holdout_test_dataset.targets),
        ),
    }

    print("=" * 60)
    print("DATA PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Image size      : {IMG_SIZE}×{IMG_SIZE}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Classes         : {CLASS_NAMES}")
    print(f"Total (original): {len(full_dataset)} training + "
          f"{len(holdout_test_dataset)} held-out test")
    print("-" * 60)

    for name, (indices, targets) in splits.items():
        counts = Counter(targets[indices])
        dist   = "  |  ".join(
            f"{CLASS_NAMES[k]}: {counts[k]}" for k in sorted(counts)
        )
        print(f"{name:25s} — {len(indices):>5} images  [{dist}]")

    print("=" * 60)


# ──────────────────────────────────────────────
# 7. VISUALISATION HELPER
# ──────────────────────────────────────────────
def show_batch(loader, num_images=9, title="Sample Training Batch"):
    """
    Fetch one batch from *loader* and display `num_images` in a 3×3 grid.

    Each subplot title shows the human-readable class label so you can
    visually verify the pipeline is loading and labelling images correctly.

    Because images are normalised (ImageNet stats), we reverse the
    normalisation before plotting so the pixel values look natural.
    """
    # Grab a single batch.
    images, labels = next(iter(loader))

    # Reverse the ImageNet normalisation for display purposes.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_denorm = images[:num_images] * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)    # keep in [0, 1]

    # Plot a 3×3 grid.
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx < num_images and idx < len(labels):
            # Convert from [C, H, W] → [H, W, C] for matplotlib.
            img_np = images_denorm[idx].permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(CLASS_NAMES[labels[idx]], fontsize=12, color="white",
                         backgroundcolor="black", pad=6)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "sample_batch.png"), dpi=150,
                bbox_inches="tight")
    print(f"\n✓ Sample batch saved to {os.path.join(BASE_DIR, 'sample_batch.png')}")
    plt.show()


# ──────────────────────────────────────────────
# 8. DEVICE DETECTION
# ──────────────────────────────────────────────
def get_device():
    """Select the best available compute device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ──────────────────────────────────────────────
# 9. MAIN — run when script is executed directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    print_split_summary()
    show_batch(train_loader)
