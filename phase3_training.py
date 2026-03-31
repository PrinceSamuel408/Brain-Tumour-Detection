"""
Phase 3: Training Loop & Loss Function
=======================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script implements the complete training pipeline:

    1. Loss function   — CrossEntropyLoss (standard for multi-class tasks).
    2. Optimizer       — Adam with lr=1e-4, applied ONLY to trainable params
                         (the custom classifier head).
    3. Training loop   — per-epoch train + validation with metrics tracking.
    4. Early stopping  — halts training if val loss stagnates for 5 epochs.
    5. Checkpointing   — saves model weights only when val loss improves.
    6. Visual tracking — loss & accuracy curves plotted after training.

Why Adam at 1e-4?
    - Adam adapts per-parameter learning rates, which is ideal when only a
      small head is trainable atop a frozen backbone.
    - 1e-4 is conservative enough to avoid overshooting the loss landscape
      of a pretrained network, but large enough to converge in reasonable time.

Why CrossEntropyLoss?
    - It combines LogSoftmax + NLLLoss in one numerically stable operation.
    - Expects raw logits (no softmax needed in the model's forward pass).
    - Our 4 classes are balanced, so no class-weight adjustment is needed.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Phase 1 — data loaders, class names, device helper.
from phase1_data_pipeline import (
    train_loader,
    val_loader,
    get_device,
    CLASS_NAMES,
)

# Phase 2 — model builder.
from phase2_model_architecture import build_model, count_parameters

# ──────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH  = os.path.join(BASE_DIR, "best_model.pth")

# Training hyperparameters
NUM_EPOCHS       = 30         # maximum epochs (early stopping may cut this short)
LEARNING_RATE    = 1e-4       # conservative LR for fine-tuning
PATIENCE         = 5          # early-stopping patience (epochs without improvement)


# ──────────────────────────────────────────────
# 3. SINGLE-EPOCH TRAINING STEP
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass over the training set.

    Steps per batch:
        1. Move images & labels to device.
        2. Zero the gradient buffers (otherwise gradients accumulate).
        3. Forward pass — compute logits.
        4. Compute loss (CrossEntropyLoss on raw logits).
        5. Backward pass — compute gradients w.r.t. trainable parameters.
        6. Optimizer step — update weights.
        7. Accumulate running loss and correct-prediction count.

    Returns
    ───────
    epoch_loss : float   — average loss over all batches
    epoch_acc  : float   — accuracy (correct / total) over the full epoch
    """
    model.train()                              # enable dropout & batch-norm training mode

    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    for images, labels in loader:
        # ── Move data to device ──────────────────────────
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ─────────────────────────────────
        optimizer.zero_grad()                  # clear stale gradients
        logits = model(images)                 # raw scores, shape (B, 4)
        loss   = criterion(logits, labels)     # scalar loss

        # ── Backward pass + weight update ────────────────
        loss.backward()                        # compute gradients
        optimizer.step()                       # update trainable parameters

        # ── Accumulate metrics ───────────────────────────
        running_loss    += loss.item() * images.size(0)    # sum of per-sample losses
        _, preds         = torch.max(logits, dim=1)        # predicted class indices
        running_correct += (preds == labels).sum().item()
        running_total   += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc  = running_correct / running_total
    return epoch_loss, epoch_acc


# ──────────────────────────────────────────────
# 4. SINGLE-EPOCH VALIDATION STEP
# ──────────────────────────────────────────────
@torch.no_grad()                               # globally disable gradient tracking
def validate_one_epoch(model, loader, criterion, device):
    """
    Evaluate the model on the validation set WITHOUT computing gradients.

    torch.no_grad() is critical here:
        - Saves GPU memory (no computation graph stored).
        - Prevents accidental weight updates.
        - Speeds up inference by ~20-30 %.

    Returns
    ───────
    epoch_loss : float
    epoch_acc  : float
    """
    model.eval()                               # disable dropout & freeze batch-norm stats

    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss    += loss.item() * images.size(0)
        _, preds         = torch.max(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        running_total   += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc  = running_correct / running_total
    return epoch_loss, epoch_acc


# ──────────────────────────────────────────────
# 5. TRAINING HISTORY PLOTTER
# ──────────────────────────────────────────────
def plot_training_history(history, save_path=None):
    """
    Plot Training vs Validation curves for both Loss and Accuracy.

    Generates a side-by-side figure:
        Left panel  — loss curves   (lower is better)
        Right panel — accuracy curves (higher is better)

    A vertical dashed line marks the epoch with the best val loss (i.e.,
    the checkpoint that was saved).

    Parameters
    ──────────
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    save_path : str or None — if provided, saves the figure to disk.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Find the epoch with the best validation loss (1-indexed for display).
    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # ── Left panel: Loss ─────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right panel: Accuracy ────────────────────────────
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=4, label="Val Acc")
    ax2.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Training curves saved to {save_path}")

    plt.show()


# ──────────────────────────────────────────────
# 6. MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    checkpoint_path=CHECKPOINT_PATH,
):
    """
    Full training loop with early stopping and model checkpointing.

    Parameters
    ──────────
    model           : nn.Module  — the BrainTumorClassifier from Phase 2.
    train_loader    : DataLoader — training batches (shuffled, augmented).
    val_loader      : DataLoader — validation batches (deterministic).
    device          : torch.device
    num_epochs      : int        — maximum number of epochs.
    lr              : float      — learning rate for Adam.
    patience        : int        — early-stopping patience.
    checkpoint_path : str        — where to save the best model weights.

    Returns
    ───────
    model   : nn.Module — with the best weights loaded back in.
    history : dict      — per-epoch train/val loss and accuracy.
    """
    # ── Loss function ────────────────────────────────────
    # CrossEntropyLoss expects:
    #   • input:  raw logits of shape (B, C)   — no softmax needed
    #   • target: class indices of shape (B,)  — integers in [0, C-1]
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────
    # CRITICAL: only pass parameters where requires_grad=True.
    # This ensures Adam only tracks and updates the classifier head,
    # not the 23 M frozen backbone parameters (saves memory + speed).
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr)

    # ── Tracking variables ───────────────────────────────
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    best_val_loss    = float("inf")        # initialise to worst possible
    best_model_wts   = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0                  # early-stopping counter
    total_start_time  = time.time()

    # ── Print training config ────────────────────────────
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device          : {device}")
    print(f"Max epochs      : {num_epochs}")
    print(f"Learning rate   : {lr}")
    print(f"Patience        : {patience}")
    print(f"Loss function   : CrossEntropyLoss")
    print(f"Optimizer       : Adam")
    print(f"Checkpoint path : {checkpoint_path}")
    print("=" * 60)

    # ── Epoch loop ───────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train for one epoch.
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate.
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Record metrics.
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        # ── Print epoch summary ──────────────────────────
        print(
            f"Epoch [{epoch:02d}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  │  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  │  "
            f"Time: {epoch_time:.1f}s",
            end="",
        )

        # ── Checkpoint: save if val loss improved ────────
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save checkpoint to disk.
            torch.save(best_model_wts, checkpoint_path)
            print(f"  ✓ Saved (↓{improvement:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  — No improve ({epochs_no_improve}/{patience})")

        # ── Early stopping check ─────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {patience} consecutive epochs).")
            break

    # ── Training complete ────────────────────────────────
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time        : {total_time / 60:.1f} minutes")
    print(f"Best val loss     : {best_val_loss:.4f}")
    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
    print(f"Best epoch        : {best_epoch}")
    print(f"Best val accuracy : {history['val_acc'][best_epoch - 1]:.4f}")
    print(f"Checkpoint saved  : {checkpoint_path}")
    print("=" * 60)

    # Load the best weights back into the model before returning.
    model.load_state_dict(best_model_wts)

    return model, history


# ──────────────────────────────────────────────
# 7. MAIN — execute training
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Detect compute device.
    device = get_device()

    # Build model (Phase 2) and move to device.
    model = build_model(device)
    count_parameters(model)

    # Run the training loop.
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Plot and save training curves.
    plot_training_history(
        history,
        save_path=os.path.join(BASE_DIR, "training_curves.png"),
    )
