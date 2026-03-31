# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Interactive Web Application — Streamlit
# ──────────────────────────────────────────────────────────────────────────────
# Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch
#
# HOW TO RUN (do NOT use `python phase6_app.py`):
# ───────────────────────────────────────────────
#   cd "/Users/princesamuel/Brain Tumour Antigravity/Brain-Tumour-Detection"
#   streamlit run phase6_app.py
#
# This will start a local server (usually http://localhost:8501) and open
# the app in your default browser.
#
# DEPENDENCIES:
#   pip install streamlit opencv-python
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F_torch
from PIL import Image
from torchvision import transforms

# ──────────────────────────────────────────────
# Ensure the project root is on sys.path so
# sibling-module imports (phase1, phase2, …) work
# regardless of where Streamlit is launched from.
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from phase2_model_architecture import build_model          # model constructor
from phase5_gradcam import GradCAM, denormalize_image, overlay_heatmap  # XAI utils
from phase1_data_pipeline import CLASS_NAMES, IMG_SIZE, get_device      # constants


# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model.pth")

# Human-friendly display names for the sidebar.
CLASS_DISPLAY = {
    "glioma":      "🔴 Glioma — aggressive tumor arising from glial cells",
    "meningioma":  "🟠 Meningioma — usually benign, growing from the meninges",
    "notumor":     "🟢 No Tumor — healthy brain scan with no visible mass",
    "pituitary":   "🔵 Pituitary — tumor of the pituitary gland at the skull base",
}

# ImageNet normalisation stats (must match Phase 1 transforms).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Preprocessing pipeline — identical to Phase 1's eval_transform.
# We define it here explicitly so the Streamlit app is self-contained
# and does not depend on Phase 1's module-level dataset loading.
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────
# 2. MODEL LOADING (cached across reruns)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    """
    Build the BrainTumorClassifier, load trained weights, and prepare
    the Grad-CAM hook.

    @st.cache_resource ensures this runs ONCE — the model persists in
    memory across every user interaction and page rerun.

    Returns
    ───────
    model    : nn.Module in eval mode, on the best available device.
    grad_cam : GradCAM instance with hooks on layer4[-1].
    device   : torch.device used.
    """
    device = get_device()
    model  = build_model(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Set up Grad-CAM on the last bottleneck block of ResNet50.
    target_layer = model.backbone.layer4[-1]
    grad_cam     = GradCAM(model, target_layer)

    return model, grad_cam, device


# ──────────────────────────────────────────────
# 3. PREDICTION PIPELINE
# ──────────────────────────────────────────────
def predict(image: Image.Image, model, grad_cam, device):
    """
    Run the full prediction + explainability pipeline on a single PIL image.

    Steps
    ─────
    1. Preprocess (resize, grayscale→3ch, normalize) — same as Phase 1.
    2. Forward pass → raw logits.
    3. Softmax → confidence percentages.
    4. Grad-CAM → heatmap overlay.

    Parameters
    ──────────
    image    : PIL.Image — the user-uploaded MRI scan.
    model    : nn.Module — the trained classifier.
    grad_cam : GradCAM   — hook-equipped explainer.
    device   : torch.device.

    Returns
    ───────
    predicted_class : str   — human-readable class name.
    confidences     : dict  — {class_name: percentage} for all 4 classes.
    original_np     : np.ndarray (H, W, 3) [0,1] — denormalised input.
    overlay_np      : np.ndarray (H, W, 3) [0,1] — Grad-CAM overlay.
    """
    # ── Step 1: Preprocess ───────────────────────────────
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # ── Step 2: Forward pass + softmax ───────────────────
    with torch.no_grad():
        logits = model(input_tensor)                          # (1, 4)
    probabilities = F_torch.softmax(logits, dim=1).squeeze()  # (4,)

    predicted_idx   = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]

    confidences = {
        CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    # ── Step 3: Grad-CAM (requires gradients) ────────────
    # Re-create tensor WITH gradient tracking for Grad-CAM.
    input_for_cam = preprocess(image).unsqueeze(0).to(device)
    heatmap, _    = grad_cam.generate(input_for_cam, target_class=predicted_idx)

    # ── Step 4: Build overlay ────────────────────────────
    original_np = denormalize_image(input_for_cam.squeeze(0))
    overlay_np, _ = overlay_heatmap(original_np, heatmap)

    return predicted_class, confidences, original_np, overlay_np


# ──────────────────────────────────────────────
# 4. STREAMLIT UI
# ──────────────────────────────────────────────
def main():
    # ── Page config ──────────────────────────────────────
    st.set_page_config(
        page_title="Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
    )

    # ── Custom CSS for a polished look ───────────────────
    st.markdown("""
    <style>
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 1rem 0 0.5rem;
        }
        .main-header h1 {
            font-size: 2.4rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .main-header p {
            color: #888; font-size: 1.05rem;
        }

        /* Result card */
        .prediction-card {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        .prediction-card h2 {
            color: #667eea;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        .prediction-card .class-name {
            font-size: 2.2rem;
            font-weight: 700;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .prediction-card .confidence {
            font-size: 1.3rem;
            color: #a0aec0;
            margin-top: 0.3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }

        /* Image captions */
        .image-caption {
            text-align: center;
            font-weight: 600;
            font-size: 1.05rem;
            margin-top: 0.5rem;
            color: #cbd5e0;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Explainable Brain Tumor Detection</h1>
        <p>Upload a brain MRI scan to get an AI-powered diagnosis with visual explainability (Grad-CAM)</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ How It Works")
        st.markdown("""
        This application uses a **ResNet50** deep learning model fine-tuned
        on brain MRI scans to classify tumors into one of **4 categories**.

        **Pipeline:**
        1. Upload a brain MRI image
        2. The image is preprocessed (resized to 224×224, normalised)
        3. A trained CNN predicts the tumor class
        4. **Grad-CAM** highlights the regions the model focused on

        ---
        """)

        st.subheader("🏷️ The 4 Classes")
        for class_name in CLASS_NAMES:
            st.markdown(f"**{CLASS_DISPLAY[class_name]}**")

        st.markdown("---")
        st.caption(
            "Model: ResNet50 · Accuracy: ~92% · "
            "Trained on glioma, meningioma, pituitary & healthy MRI scans"
        )

    # ── Load model (cached) ──────────────────────────────
    model, grad_cam, device = load_model()

    # ── File uploader ────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📤 Upload a Brain MRI Scan",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. The image will be resized to 224×224.",
    )

    if uploaded_file is not None:
        # Open the uploaded image.
        image = Image.open(uploaded_file).convert("RGB")

        # ── Run prediction ───────────────────────────────
        with st.spinner("🔬 Analyzing MRI scan…"):
            predicted_class, confidences, original_np, overlay_np = predict(
                image, model, grad_cam, device
            )

        # ── Prediction result card ───────────────────────
        top_confidence = confidences[predicted_class]
        st.markdown(f"""
        <div class="prediction-card">
            <h2>Diagnosis Result</h2>
            <div class="class-name">{predicted_class.replace("notumor", "No Tumor")}</div>
            <div class="confidence">Confidence: {top_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar chart ─────────────────────────
        st.subheader("📊 Class Confidence Scores")

        chart_data = {
            name.replace("notumor", "No Tumor").title(): conf
            for name, conf in confidences.items()
        }
        st.bar_chart(chart_data, horizontal=True, height=200)

        # ── Two-column image display ─────────────────────
        st.subheader("🔍 Visual Explanation (Grad-CAM)")
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_np,
                caption="Original MRI Scan",
                use_container_width=True,
                clamp=True,
            )

        with col2:
            st.image(
                overlay_np,
                caption="Grad-CAM Overlay — Warm regions = high model attention",
                use_container_width=True,
                clamp=True,
            )

        # ── Interpretation guide ─────────────────────────
        with st.expander("📖 How to read the Grad-CAM overlay"):
            st.markdown("""
            The **Grad-CAM overlay** shows which regions of the MRI the model
            focused on when making its prediction:

            - 🔴 **Red / Yellow** — High attention. The model considers these
              regions most important for its classification decision.
            - 🔵 **Blue / Green** — Low attention. These areas had little
              influence on the prediction.

            **Clinical interpretation:**
            - For **tumor classes** (Glioma, Meningioma, Pituitary), you should
              see the heatmap concentrated around the tumor mass. This confirms
              the model is looking at the right anatomical region.
            - For **No Tumor**, the attention is typically diffuse (spread out),
              because the model confirms the *absence* of a focal lesion.

            > ⚠️ **Disclaimer:** This tool is for educational and research
            > purposes only. Always consult a qualified radiologist for clinical
            > diagnosis.
            """)

    else:
        # ── Empty state ──────────────────────────────────
        st.markdown("---")
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.info(
                "👆 **Upload a brain MRI scan** using the file uploader above "
                "to get started. The model will classify the scan and show you "
                "exactly which regions influenced its decision.",
                icon="🧠",
            )


# ──────────────────────────────────────────────
# 5. ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
