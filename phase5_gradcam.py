"""
Phase 5: Explainable AI — Grad-CAM (Gradient-weighted Class Activation Mapping)
=================================================================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script makes the model's decisions *interpretable* by visualising which
spatial regions of a brain MRI the model focuses on when predicting each class.

Background — Why Grad-CAM?
──────────────────────────
Deep CNNs are often criticised as "black boxes".  In medical imaging this is
unacceptable — a clinician needs to know *why* the model thinks a scan shows a
glioma.  Grad-CAM (Selvaraju et al., 2017) answers that question by producing
a coarse localisation heatmap that highlights the discriminative image regions.

Mathematical Intuition
──────────────────────
Given a target class c and the feature maps A^k from the last convolutional
layer (shape [K, H, W] where K = number of channels):

    1. Compute the gradient of the class score y^c (before softmax) w.r.t.
       each feature map:  ∂y^c / ∂A^k    (shape [K, H, W])

    2. Global-average-pool each gradient map to get a single importance
       weight per channel:

           α_k^c  =  (1 / H·W)  Σ_i Σ_j  (∂y^c / ∂A^k)_{ij}

       This tells us: "How important is channel k for predicting class c?"

    3. Compute a weighted combination of feature maps:

           L_Grad-CAM  =  ReLU( Σ_k  α_k^c · A^k )

       The ReLU keeps only features with POSITIVE influence on class c
       (negative values correspond to features important for OTHER classes).

    4. Upsample the coarse heatmap (7×7 for ResNet50's layer4) to the
       original image size (224×224), normalise to [0, 1], and overlay.

Target Layer
────────────
We hook into `model.backbone.layer4[-1]` — the very last bottleneck block
of ResNet50.  This layer produces 2048-channel, 7×7 feature maps that encode
the highest-level spatial features the network has learned.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Phase 1 — test DataLoader, class names, transforms, device helper.
from phase1_data_pipeline import (
    test_loader,
    get_device,
    CLASS_NAMES,
    IMG_SIZE,
)

# Phase 2 — model constructor.
from phase2_model_architecture import build_model


# ──────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH  = os.path.join(BASE_DIR, "best_model.pth")
OUTPUT_PATH      = os.path.join(BASE_DIR, "gradcam_output.png")

# ImageNet normalisation stats (must match Phase 1 transforms).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ──────────────────────────────────────────────
# 3. GRAD-CAM CLASS
# ──────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation using PyTorch forward & backward hooks.

    Hooks are the key mechanism here:
        • A **forward hook** fires after the target layer's forward() completes.
          We use it to capture the output feature maps (activations).
        • A **backward hook** fires after gradients are computed for the
          target layer during .backward().  We use it to capture the
          gradients of the class score w.r.t. those feature maps.

    Parameters
    ──────────
    model        : nn.Module — the trained BrainTumorClassifier.
    target_layer : nn.Module — the specific layer to attach hooks to.
                   For ResNet50 this is model.backbone.layer4[-1].
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer

        # Storage for captured tensors.
        self.activations = None   # feature maps from forward pass
        self.gradients   = None   # gradients from backward pass

        # ── Register hooks ───────────────────────────────
        # forward_hook: called after target_layer.forward()
        #   → saves the layer's output tensor (activations).
        self._forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        # full_backward_hook: called after gradients are computed for
        # the target layer during loss.backward()
        #   → saves grad_output (the gradient of the loss w.r.t. the
        #     layer's output).
        self._backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, input, output):
        """
        Capture output activations from the target convolutional layer.

        CRITICAL: We call output.retain_grad() here.  Without this,
        PyTorch would NOT compute gradients for intermediate tensors
        produced by frozen layers (requires_grad=False on the weights).
        retain_grad() tells PyTorch: "Even though this tensor wasn't
        created by a leaf variable requiring gradients, I still want
        its gradient to be stored after .backward()."
        """
        self.activations = output

        # During regular inference (torch.no_grad), this tensor will not
        # track gradients. Only retain grad when autograd is active.
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.retain_grad()

    def _backward_hook(self, module, grad_input, grad_output):
        """
        Capture the gradient of the target class score w.r.t. the
        feature maps A.  grad_output is a tuple; the first element
        has shape (B, K, H, W) — same shape as the activations.

        These gradients encode: "How much would the class score change
        if we slightly changed each spatial location in each channel?"
        """
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        """
        Generate the Grad-CAM heatmap for a single image.

        Steps
        ─────
        1. Forward pass → get logits + activations (via hook).
        2. If no target_class given, use the predicted class.
        3. Backward pass on the target class score → get gradients (via hook).
        4. Global-average-pool gradients → channel importance weights α_k.
        5. Weighted sum of activations → coarse heatmap.
        6. ReLU → only positive influence.
        7. Normalise to [0, 1].

        Parameters
        ──────────
        input_tensor : Tensor of shape (1, 3, 224, 224) — single preprocessed image.
        target_class : int or None — class index to explain.  If None, explains
                       the model's top prediction.

        Returns
        ───────
        heatmap      : np.ndarray of shape (224, 224), values in [0, 1].
        predicted    : int — the class index the model predicted.
        """
        self.model.eval()

        # Ensure the input requires gradients so the backward pass
        # can propagate through it to the target layer.
        input_tensor = input_tensor.requires_grad_(True)

        # ── Step 1: Forward pass ─────────────────────────
        logits = self.model(input_tensor)          # (1, num_classes)
        predicted = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted

        # ── Step 2: Backward pass ────────────────────────
        # Zero all existing gradients to prevent accumulation from
        # previous calls.
        self.model.zero_grad()

        # Select the target class score (a single scalar).
        # .backward() computes ∂y^c / ∂A for every tensor in the graph.
        target_score = logits[0, target_class]
        target_score.backward()

        # ── Step 3: Compute Grad-CAM ────────────────────
        # gradients shape: (1, K, H, W) where K=2048, H=W=7 for ResNet50
        gradients   = self.gradients.detach()      # (1, 2048, 7, 7)
        activations = self.activations.detach()    # (1, 2048, 7, 7)

        # Global average pooling of gradients → importance weights.
        # α_k = (1 / H·W) Σ_i Σ_j  (∂y^c / ∂A^k)_{ij}
        # Result shape: (1, 2048, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination: Σ_k  α_k · A^k
        # Multiply each channel by its weight, then sum over channels.
        # Result shape: (1, 1, 7, 7)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU: keep only features with POSITIVE influence on class c.
        # Negative values = features that suppress this class (irrelevant
        # for explaining *why* the model chose this class).
        cam = F.relu(cam)

        # ── Step 4: Upsample to image size ──────────────
        cam = F.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),             # 224 × 224
            mode="bilinear",
            align_corners=False,
        )

        # Squeeze to 2D: (224, 224)
        cam = cam.squeeze().cpu().numpy()

        # ── Step 5: Normalise to [0, 1] ─────────────────
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:               # avoid division by zero
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted

    def remove_hooks(self):
        """Remove the registered hooks to free memory."""
        self._forward_handle.remove()
        self._backward_handle.remove()


# ──────────────────────────────────────────────
# 4. VISUALISATION UTILITIES
# ──────────────────────────────────────────────
def denormalize_image(tensor):
    """
    Reverse the ImageNet normalisation applied in Phase 1 so the image
    looks natural when displayed.

    Input  : Tensor of shape (3, 224, 224) — normalised.
    Output : np.ndarray of shape (224, 224, 3) — pixel values in [0, 1].
    """
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN       # reverse normalisation
    img = np.clip(img, 0, 1)                       # clamp to valid range
    return img


def overlay_heatmap(image_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Blend a Grad-CAM heatmap onto the original image.

    Parameters
    ──────────
    image_np  : np.ndarray (H, W, 3), float [0, 1] — original MRI scan.
    heatmap   : np.ndarray (H, W),    float [0, 1] — Grad-CAM activation map.
    alpha     : float   — blending weight for the heatmap (0 = only image,
                          1 = only heatmap).
    colormap  : int     — OpenCV colormap ID (cv2.COLORMAP_JET gives the
                          classic red/blue thermal look).

    Returns
    ───────
    overlay : np.ndarray (H, W, 3), float [0, 1] — blended image.
    colored_heatmap : np.ndarray (H, W, 3), float [0, 1] — coloured heatmap only.
    """
    # Convert heatmap to uint8 for OpenCV colourmap.
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply the JET colourmap → gives a (H, W, 3) BGR image.
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR → RGB and normalise back to [0, 1].
    colored_heatmap = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB) / 255.0

    # Alpha-blend: overlay = α · heatmap + (1-α) · original.
    overlay = alpha * colored_heatmap + (1 - alpha) * image_np
    overlay = np.clip(overlay, 0, 1)

    return overlay, colored_heatmap


# ──────────────────────────────────────────────
# 5. FIND ONE CORRECT SAMPLE PER CLASS
# ──────────────────────────────────────────────
def find_correct_samples(model, device):
    """
    Scan the test set to find one correctly classified sample from EACH
    of the 4 classes.

    Returns a dict:  { class_idx: (image_tensor, true_label) }
    where image_tensor has shape (1, 3, 224, 224).
    """
    model.eval()
    found = {}     # class_idx → (image_tensor, label)
    needed = set(range(len(CLASS_NAMES)))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)

            for i in range(images.size(0)):
                true_label = labels[i].item()
                pred_label = preds[i].item()

                # Only keep correctly classified samples.
                if true_label == pred_label and true_label in needed:
                    # Store as a single-image batch (1, 3, 224, 224).
                    found[true_label] = (
                        images[i].unsqueeze(0).clone(),
                        true_label,
                    )
                    needed.discard(true_label)
                    print(f"  ✓ Found correct sample for class "
                          f"'{CLASS_NAMES[true_label]}' (index {true_label})")

                if not needed:
                    return found

    # If some classes weren't found (unlikely with 92 % acc), warn.
    if needed:
        missing = [CLASS_NAMES[c] for c in needed]
        print(f"  ⚠ Could not find correct samples for: {missing}")

    return found


# ──────────────────────────────────────────────
# 6. CREATE THE 4×3 GRAD-CAM VISUALISATION GRID
# ──────────────────────────────────────────────
def create_gradcam_grid(model, device, save_path=OUTPUT_PATH):
    """
    Produce a publication-quality 4×3 figure:
        Row  = one class (glioma, meningioma, notumor, pituitary)
        Col1 = Original MRI
        Col2 = Grad-CAM heatmap (standalone)
        Col3 = Overlay (heatmap blended onto original)

    The figure is saved to *save_path* and displayed.
    """
    # ── Find one correct sample per class ────────────────
    print("\nSearching for correctly classified samples (one per class)…")
    samples = find_correct_samples(model, device)

    if len(samples) < len(CLASS_NAMES):
        print("⚠ Not all classes have a correct sample — grid will be partial.")

    # ── Set up Grad-CAM ──────────────────────────────────
    # Target: last bottleneck block of ResNet50's layer4.
    target_layer = model.backbone.layer4[-1]
    grad_cam     = GradCAM(model, target_layer)

    # ── Build the figure ─────────────────────────────────
    num_classes = len(samples)
    fig, axes   = plt.subplots(num_classes, 3, figsize=(14, 4.5 * num_classes))

    # Ensure axes is 2D even if only one row.
    if num_classes == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Grad-CAM Explanations — Brain Tumor Classification\n"
        "(What is the model looking at for each class?)",
        fontsize=16, fontweight="bold", y=1.01,
    )

    col_titles = ["Original MRI", "Grad-CAM Heatmap", "Overlay"]

    for row_idx, class_idx in enumerate(sorted(samples.keys())):
        image_tensor, true_label = samples[class_idx]

        # ── Generate Grad-CAM heatmap ────────────────────
        heatmap, pred = grad_cam.generate(
            image_tensor.to(device),
            target_class=class_idx,
        )

        # ── Prepare images ───────────────────────────────
        original     = denormalize_image(image_tensor.squeeze(0))
        overlay, colored_heatmap = overlay_heatmap(original, heatmap)

        # ── Plot ─────────────────────────────────────────
        panels = [original, colored_heatmap, overlay]
        for col_idx, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.axis("off")

            # Column titles on the top row only.
            if row_idx == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Row label (class name) on the left.
        axes[row_idx, 0].set_ylabel(
            CLASS_NAMES[class_idx].upper(),
            fontsize=13, fontweight="bold",
            rotation=90, labelpad=15,
        )
        # Re-enable the y-axis label but keep ticks off.
        axes[row_idx, 0].yaxis.set_visible(True)
        axes[row_idx, 0].set_yticks([])

    # ── Save and show ────────────────────────────────────
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ Grad-CAM grid saved to {save_path}")
    plt.show()

    # Clean up hooks.
    grad_cam.remove_hooks()


# ──────────────────────────────────────────────
# 7. MAIN — run the full Grad-CAM pipeline
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: Device & model ───────────────────────────
    device = get_device()
    model  = build_model(device)

    # Load trained weights.
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model loaded from {CHECKPOINT_PATH}")

    # ── Step 2: Generate and save Grad-CAM grid ─────────
    create_gradcam_grid(model, device)

    print("\n✓ Phase 5 (Grad-CAM explainability) complete.")
