"""
Phase 2: Model Architecture Setup
==================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script defines the classifier architecture using transfer learning:

    1. Load a ResNet50 backbone pretrained on ImageNet (1.2 M natural images).
    2. Freeze all convolutional (feature-extraction) layers so that ImageNet
       knowledge is preserved during early training.
    3. Replace the final fully-connected head with a lightweight custom
       classifier tuned for our 4-class brain tumor task.

Why ResNet50?
    - Deep enough (50 layers) to capture complex texture / shape features in
      MRI scans, but not so large that it is impractical on a single GPU.
    - Skip connections prevent vanishing gradients, which is critical when
      we later unfreeze layers for fine-tuning.
    - The 2048-dimensional feature vector from its penultimate layer provides
      a rich, general-purpose image representation.

Why freeze first?
    - Our dataset (≈ 5 600 images) is small relative to ImageNet.  Training
      23 M+ convolutional parameters from scratch would overfit rapidly.
    - Freezing lets only the new classifier head learn, acting as a powerful
      feature extractor with a lightweight trainable head on top.
    - In later phases we can selectively unfreeze deeper layers for fine-tuning
      once the head has converged.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

# Re-use the device helper from Phase 1.
from phase1_data_pipeline import get_device, CLASS_NAMES, IMG_SIZE


# ──────────────────────────────────────────────
# 2. MODEL DEFINITION
# ──────────────────────────────────────────────
class BrainTumorClassifier(nn.Module):
    """
    Transfer-learning classifier for brain MRI tumour detection.

    Architecture
    ─────────────
    ┌─────────────────────────────────────────────┐
    │  ResNet50 backbone (pretrained, FROZEN)      │
    │  conv1 → bn1 → relu → maxpool               │
    │  layer1  (3 bottleneck blocks)               │
    │  layer2  (4 bottleneck blocks)               │
    │  layer3  (6 bottleneck blocks)               │
    │  layer4  (3 bottleneck blocks)               │
    │  avgpool → flatten  → 2048-d feature vector  │
    └──────────────────────┬──────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────┐
    │  Custom classifier head (TRAINABLE)          │
    │  Linear(2048, 256) → ReLU → Dropout(0.5)    │
    │  Linear(256, 4)                              │
    └─────────────────────────────────────────────┘

    Parameters
    ──────────
    num_classes : int
        Number of output classes (default 4: glioma, meningioma, notumor,
        pituitary).
    dropout_rate : float
        Dropout probability in the classifier head (default 0.5).
        Medical datasets are small, so aggressive dropout helps regularise.
    """

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super().__init__()

        # ── 2a. Load the pretrained ResNet50 backbone ──────────────
        # ResNet50_Weights.DEFAULT loads the best available weights
        # (currently IMAGENET1K_V2, top-1 acc ≈ 80.9 %).
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # ── 2b. Freeze every parameter in the backbone ─────────────
        # requires_grad = False prevents gradient computation and weight
        # updates for all convolutional & batch-norm layers.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── 2c. Capture the input dimension of the original fc layer ─
        # ResNet50's final fc is Linear(2048, 1000).  We need the
        # in_features value (2048) to build our replacement head.
        in_features = self.backbone.fc.in_features   # 2048

        # ── 2d. Replace the fc layer with our custom head ──────────
        # The new head is intentionally simple:
        #   • 2048 → 256  reduces dimensionality (fewer params → less overfit)
        #   • ReLU        introduces non-linearity
        #   • Dropout     randomly zeros 50 % of activations during training
        #   • 256 → 4     maps to our target classes
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        # After replacing fc, unfreeze the new head's parameters so they
        # can be trained.  (They were not part of the original backbone
        # when we froze everything, so they are already unfrozen — but
        # being explicit here makes the intent clear and future-proof.)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ──────────
        x : Tensor of shape (B, 3, 224, 224)
            A batch of preprocessed RGB brain MRI images.

        Returns
        ───────
        Tensor of shape (B, num_classes)
            Raw logits (unnormalised scores) for each class.
            Pass through softmax / argmax downstream for probabilities / labels.
        """
        return self.backbone(x)


# ──────────────────────────────────────────────
# 3. HELPER UTILITIES
# ──────────────────────────────────────────────
def count_parameters(model: nn.Module):
    """
    Count and print total vs trainable parameters.

    This verifies that the backbone is frozen (millions of params with
    requires_grad=False) and only the custom head is trainable.
    """
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print("=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total parameters      : {total_params:>12,}")
    print(f"Trainable parameters  : {trainable_params:>12,}")
    print(f"Frozen parameters     : {frozen_params:>12,}")
    print(f"Trainable ratio       : {100 * trainable_params / total_params:.2f} %")
    print("=" * 60)

    return total_params, trainable_params


def verify_forward_pass(model: nn.Module, device: torch.device):
    """
    Pass a dummy tensor through the model and assert the output shape is
    correct.  This catches shape mismatches early, before we start a long
    training run.
    """
    model.eval()                                        # disable dropout for this test
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    with torch.no_grad():                               # no gradients needed for a shape check
        output = model(dummy_input)

    expected_shape = (1, len(CLASS_NAMES))               # (1, 4)
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    print(f"\n✓ Forward pass verified")
    print(f"  Input shape  : {tuple(dummy_input.shape)}")
    print(f"  Output shape : {tuple(output.shape)}  (matches {expected_shape})")
    print(f"  Output logits: {output.cpu().squeeze().tolist()}")


def print_model_architecture(model: nn.Module):
    """Print the custom classifier head so we can inspect layer details."""
    print("\n" + "=" * 60)
    print("CUSTOM CLASSIFIER HEAD  (backbone.fc)")
    print("=" * 60)
    for name, layer in model.backbone.fc.named_children():
        print(f"  [{name}] {layer}")
    print("=" * 60)


# ──────────────────────────────────────────────
# 4. BUILD FUNCTION (for importing in later phases)
# ──────────────────────────────────────────────
def build_model(device: torch.device, num_classes: int = 4) -> BrainTumorClassifier:
    """
    Convenience function to instantiate the model and move it to the target
    device in one call.  Used by Phase 3 (training loop) and beyond.
    """
    model = BrainTumorClassifier(num_classes=num_classes)
    model = model.to(device)
    return model


# ──────────────────────────────────────────────
# 5. MAIN — verification script
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Detect device (reuses Phase 1 helper).
    device = get_device()

    # Build the model and move to device.
    model = build_model(device)

    # Show the classifier head architecture.
    print_model_architecture(model)

    # Count parameters — proves the backbone is frozen.
    count_parameters(model)

    # Run a dummy forward pass — proves the output shape is (1, 4).
    verify_forward_pass(model, device)
