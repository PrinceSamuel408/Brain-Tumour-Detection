"""
Phase 4: Detailed Model Evaluation
====================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script performs a full quantitative evaluation of the trained model
on the held-out test split (15 % of the original Training data).

Outputs
───────
1. Classification Report (console)
       Per-class Precision, Recall, F1-Score, and Support.

2. Normalised Confusion Matrix (confusion_matrix.png)
       A heatmap showing the percentage of samples from each true class
       that were predicted as each class.

How to read the confusion matrix
────────────────────────────────
• Rows    = TRUE (actual) class labels.
• Columns = PREDICTED class labels.
• Diagonal cells   → correct predictions (true positives for each class).
                       Higher diagonal values = better model performance.
• Off-diagonal cells → misclassifications.
       - Reading across a ROW shows how a given true class was (mis)classified.
         e.g. Row "glioma", Column "meningioma" = percentage of glioma images
              the model wrongly labelled as meningioma.
       - Reading down a COLUMN shows which true classes contributed false
         positives to a given predicted class.

Normalisation is done per row (row sums = 100 %) so you can compare
classes even if they have different sample counts.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import classification_report, confusion_matrix

# Phase 1 — test DataLoader, class names, device helper.
from phase1_data_pipeline import test_loader, get_device, CLASS_NAMES

# Phase 2 — model constructor.
from phase2_model_architecture import build_model


# ──────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH  = os.path.join(BASE_DIR, "best_model.pth")
CM_SAVE_PATH     = os.path.join(BASE_DIR, "confusion_matrix.png")


# ──────────────────────────────────────────────
# 3. LOAD TRAINED MODEL
# ──────────────────────────────────────────────
def load_trained_model(device: torch.device) -> torch.nn.Module:
    """
    Instantiate the BrainTumorClassifier architecture (Phase 2) and load
    the best weights saved during Phase 3 training.

    Parameters
    ──────────
    device : torch.device
        Target device (cuda / mps / cpu).

    Returns
    ───────
    model : BrainTumorClassifier with trained weights, in eval mode.
    """
    model = build_model(device)

    # Load the checkpoint.
    # map_location ensures the weights land on the correct device even if
    # the model was trained on a different one (e.g. GPU → CPU).
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Switch to evaluation mode → disables dropout and freezes batch-norm.
    model.eval()

    print(f"✓ Model loaded from {CHECKPOINT_PATH}")
    print(f"  Device: {device}")
    return model


# ──────────────────────────────────────────────
# 4. COLLECT PREDICTIONS
# ──────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run the model over every batch in *loader* and collect the true labels
    and predicted labels into flat NumPy arrays.

    Decorating with @torch.no_grad() ensures:
        • No computation graph is built  → saves GPU memory.
        • No gradient buffers are stored → faster inference.

    Parameters
    ──────────
    model  : nn.Module in eval mode.
    loader : DataLoader (test split).
    device : torch.device.

    Returns
    ───────
    all_labels : np.ndarray of shape (N,) — ground-truth class indices.
    all_preds  : np.ndarray of shape (N,) — predicted class indices.
    """
    all_labels = []
    all_preds  = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)                       # (B, 4) raw scores
        _, preds = torch.max(logits, dim=1)          # predicted class index

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    # Quick sanity check.
    total    = len(all_labels)
    correct  = (all_labels == all_preds).sum()
    accuracy = correct / total
    print(f"\n✓ Predictions collected: {total} samples, "
          f"overall accuracy = {accuracy:.4f} ({correct}/{total})")

    return all_labels, all_preds


# ──────────────────────────────────────────────
# 5. CLASSIFICATION REPORT
# ──────────────────────────────────────────────
def print_classification_report(y_true, y_pred):
    """
    Print a detailed per-class classification report.

    Metrics explained
    ─────────────────
    Precision : Of all samples the model PREDICTED as class X, what fraction
                truly belonged to class X?
                High precision → few false positives.

    Recall    : Of all samples that truly ARE class X, what fraction did the
                model correctly identify?
                High recall → few false negatives.

    F1-Score  : Harmonic mean of Precision and Recall.
                Balances both concerns; especially useful when class sizes differ.

    Support   : Number of samples of that class in the test set.
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT  (Test Split — 15 %)")
    print("=" * 60)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,                   # 4 decimal places for precision
    )
    print(report)
    print("=" * 60)

    return report


# ──────────────────────────────────────────────
# 6. NORMALISED CONFUSION MATRIX HEATMAP
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path=CM_SAVE_PATH):
    """
    Compute the confusion matrix, normalise it by row (true label), and plot
    a publication-quality heatmap using seaborn.

    Normalisation
    ─────────────
    Each row is divided by its sum so that cell values represent
    *percentages* rather than raw counts.  This makes it easy to compare
    classes that have different numbers of samples.

    Reading the result
    ──────────────────
    • Diagonal cells (top-left → bottom-right):
          Percentage of correctly classified samples for each class.
          e.g. if "glioma" row / "glioma" column = 0.93 → the model
          correctly identified 93 % of glioma images.

    • Off-diagonal cells:
          Misclassification patterns.
          e.g. "glioma" row / "meningioma" column = 0.05 → 5 % of true
          glioma images were incorrectly predicted as meningioma.

    Parameters
    ──────────
    y_true    : array-like — true class indices.
    y_pred    : array-like — predicted class indices.
    save_path : str        — output path for the .png file.
    """
    # ── Compute raw confusion matrix ─────────────────────
    cm_raw = confusion_matrix(y_true, y_pred)

    # ── Normalise per row (each row sums to 1.0) ────────
    cm_normalised = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    # ── Plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_normalised,
        annot=True,                        # show numeric values in each cell
        fmt=".2%",                         # display as percentage (e.g. 93.21 %)
        cmap="Blues",                       # blue colour gradient
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,                    # thin white gridlines
        linecolor="white",
        cbar_kws={"label": "Proportion"},  # colour-bar label
        square=True,                       # square cells
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Label",      fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title(
        "Normalised Confusion Matrix — Brain Tumor Classification\n"
        "(values = proportion of each true class predicted as each label)",
        fontsize=14, fontweight="bold", pad=16,
    )

    # Rotate tick labels for readability.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ Confusion matrix saved to {save_path}")
    plt.show()

    # ── Also print raw counts for reference ──────────────
    print("\nRaw confusion matrix (counts):")
    header = "            " + "  ".join(f"{c:>12}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm_raw):
        row_str = "  ".join(f"{v:>12}" for v in row)
        print(f"{CLASS_NAMES[i]:>12}  {row_str}")


# ──────────────────────────────────────────────
# 7. MAIN — run complete evaluation pipeline
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: Device & model ───────────────────────────
    device = get_device()
    model  = load_trained_model(device)

    # ── Step 2: Collect predictions on the test split ────
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # ── Step 3: Classification report ────────────────────
    print_classification_report(y_true, y_pred)

    # ── Step 4: Confusion matrix heatmap ─────────────────
    plot_confusion_matrix(y_true, y_pred)

    print("\n✓ Phase 4 evaluation complete.")
