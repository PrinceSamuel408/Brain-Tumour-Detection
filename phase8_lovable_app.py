"""
Phase 8: Production Streamlit App — NEUROSCAN
==============================================
Explainable Brain Tumor Detection Model — Medical Imaging with PyTorch

This script builds a polished, production-grade Streamlit interface that:
    1. Loads the trained ResNet50 model and Grad-CAM pipeline on startup.
    2. Accepts MRI uploads (PNG, JPG, JPEG).
    3. Runs real inference and returns class probabilities.
    4. Generates and displays a Grad-CAM explainability overlay.
    5. Presents results in a futuristic NEUROSCAN UI with confidence rings.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F_torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go

# Import the ML pipeline from earlier phases
from phase1_data_pipeline import CLASS_NAMES, IMG_SIZE, get_device
from phase2_model_architecture import build_model
from phase5_gradcam import GradCAM, denormalize_image, overlay_heatmap
from phase7_gatekeeper import load_gatekeeper, is_brain_mri

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model.pth")

# Preprocessing transform — must match training pipeline (Phase 1)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Page Config ---
st.set_page_config(
    page_title="NEUROSCAN — AI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

.stApp {
    background: linear-gradient(135deg, #080c14 0%, #0f1724 50%, #080c14 100%);
    color: #c8d6e5;
}

.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    text-align: center;
    color: #06b6d4;
    text-shadow: 0 0 20px rgba(6, 182, 212, 0.5), 0 0 60px rgba(6, 182, 212, 0.2);
    letter-spacing: 0.15em;
    margin-bottom: 0;
}

.main-subtitle {
    font-family: 'Inter', sans-serif;
    text-align: center;
    color: #5a6a7a;
    font-size: 0.95rem;
    letter-spacing: 0.1em;
    margin-top: 0;
}

.glass-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(200, 214, 229, 0.08);
    backdrop-filter: blur(24px);
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

.status-detected {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.75rem;
    padding: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.status-clear {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 0.75rem;
    padding: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.label-display {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: #06b6d4;
}

.scan-text {
    font-family: 'Orbitron', monospace;
    text-align: center;
    color: #06b6d4;
    text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
    letter-spacing: 0.15em;
    font-size: 0.9rem;
}

.footer-text {
    font-family: 'Inter', sans-serif;
    text-align: center;
    color: #5a6a7a;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    border-top: 1px solid rgba(200, 214, 229, 0.08);
    padding-top: 0.75rem;
    margin-top: 1rem;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Style file uploader */
.stFileUploader {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 1rem;
    padding: 1rem;
}

.stFileUploader:hover {
    border-color: rgba(6, 182, 212, 0.6);
    box-shadow: 0 0 30px rgba(6, 182, 212, 0.15);
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
}

/* ── 3D Brain Background: reposition the component iframe ── */
/* The brain component is the first iframe in the page */
.stMainBlockContainer > div > div:first-child {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 0 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
.stMainBlockContainer > div > div:first-child iframe {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    pointer-events: none !important;
    border: none !important;
}

/* All other content above */
.stMainBlockContainer > div > div:not(:first-child) {
    position: relative !important;
    z-index: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 3D PARTICLE BRAIN BACKGROUND (Three.js via components.html)
# ──────────────────────────────────────────────

BRAIN_BG_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
    html, body { margin:0; padding:0; overflow:hidden; background:transparent; }
    canvas { display:block; }
</style>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000, 0);
document.body.appendChild(renderer.domElement);

// ── Brain-shaped point cloud ──
// Dense sphere with brain-like deformations
const count = 12000;
const positions = new Float32Array(count * 3);
const colors = new Float32Array(count * 3);

for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    // Base spherical radius with brain-like bumps
    let r = 1.8;
    // Gyri/sulci bumps
    r += 0.12 * Math.sin(5 * phi) * Math.cos(3 * theta);
    r += 0.08 * Math.sin(7 * theta) * Math.cos(4 * phi);

    // Central longitudinal fissure (groove at top center)
    const sinTheta = Math.sin(theta);
    if (Math.abs(sinTheta) < 0.06 && phi < Math.PI * 0.55) {
        r *= 0.82;
    }

    // Slight vertical elongation — brain is taller than wide
    let x = r * Math.sin(phi) * Math.cos(theta);
    let y = r * Math.cos(phi) * 1.1;
    let z = r * Math.sin(phi) * Math.sin(theta) * 0.9;

    // Add scatter for organic feel
    const scatter = 0.08;
    x += (Math.random() - 0.5) * scatter;
    y += (Math.random() - 0.5) * scatter;
    z += (Math.random() - 0.5) * scatter;

    // Some particles slightly inside the volume for depth
    if (Math.random() < 0.3) {
        const shrink = 0.6 + Math.random() * 0.35;
        x *= shrink;
        y *= shrink;
        z *= shrink;
    }

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // White to light-cyan color palette (matching Lovable screenshot)
    const brightness = 0.6 + Math.random() * 0.4;
    const isCyan = Math.random() < 0.3;
    if (isCyan) {
        colors[i * 3]     = 0.4 * brightness;
        colors[i * 3 + 1] = 0.85 * brightness;
        colors[i * 3 + 2] = 0.95 * brightness;
    } else {
        // Bright white
        colors[i * 3]     = 0.8 * brightness;
        colors[i * 3 + 1] = 0.9 * brightness;
        colors[i * 3 + 2] = 0.95 * brightness;
    }
}

const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const mat = new THREE.PointsMaterial({
    size: 0.018, vertexColors: true, transparent: true,
    opacity: 0.85, sizeAttenuation: true,
    blending: THREE.AdditiveBlending, depthWrite: false,
});

const points = new THREE.Points(geo, mat);
scene.add(points);

// Slow rotation
(function animate() {
    requestAnimationFrame(animate);
    const t = Date.now() * 0.00015;
    points.rotation.y = t;
    points.rotation.x = Math.sin(t * 0.4) * 0.06;
    renderer.render(scene, camera);
})();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""

components.html(BRAIN_BG_HTML, height=600)


# ──────────────────────────────────────────────
# MODEL LOADING (cached so it only runs once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained ResNet50 model and Grad-CAM pipeline once."""
    device = get_device()
    model = build_model(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Set up Grad-CAM targeting the last bottleneck block of layer4
    target_layer = model.backbone.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    return model, grad_cam, device


# ──────────────────────────────────────────────
# REAL INFERENCE
# ──────────────────────────────────────────────
def run_inference(image: Image.Image, model, grad_cam, device):
    """
    Run real model inference and Grad-CAM on an uploaded MRI scan.

    Returns
    -------
    confidences : dict  — {class_name: percentage} for all classes
    predicted_class : str — the class with highest confidence
    overlay_np : np.ndarray — Grad-CAM overlay image (H, W, 3), values in [0, 1]
    elapsed : float — inference time in seconds
    """
    start_time = time.time()

    # Preprocess the uploaded image
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Forward pass for class probabilities
    with torch.no_grad():
        logits = model(input_tensor)
    probabilities = F_torch.softmax(logits, dim=1).squeeze()

    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]

    confidences = {
        CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    # Grad-CAM (requires gradient tracking — separate tensor)
    input_for_cam = preprocess(image).unsqueeze(0).to(device)
    heatmap, _ = grad_cam.generate(input_for_cam, target_class=predicted_idx)

    original_np = denormalize_image(input_for_cam.squeeze(0))
    overlay_np, _ = overlay_heatmap(original_np, heatmap)

    elapsed = time.time() - start_time
    return confidences, predicted_class, overlay_np, elapsed


def create_confidence_ring(label: str, value: float, is_detected: bool):
    """Creates a Plotly radial gauge for confidence."""
    color = "#ef4444" if is_detected else "#22c55e"
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        hole=0.75,
        marker=dict(colors=[color, "rgba(30,41,59,0.5)"]),
        textinfo="none",
        hoverinfo="skip",
        direction="clockwise",
        sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=140,
        width=140,
        annotations=[dict(
            text=f"<b>{value:.0f}%</b>",
            x=0.5, y=0.5,
            font=dict(size=18, color="#c8d6e5", family="Orbitron"),
            showarrow=False,
        )],
    )
    return fig


# ──────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────
st.markdown('<h1 class="main-title">NEURO<span style="color:#c8d6e5">SCAN</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">AI-Powered Brain Tumor Detection System</p>', unsafe_allow_html=True)
st.markdown("")

# Load models on startup
model, grad_cam, device = load_model()

@st.cache_resource
def load_gatekeeper_model():
    """Load the MRI gatekeeper model once."""
    d = get_device()
    return load_gatekeeper(d), d

gatekeeper_model, gatekeeper_device = load_gatekeeper_model()

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop MRI scan here — PNG, JPG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p style="text-align:center;color:#5a6a7a;font-family:Inter;font-size:0.8rem;">'
        'Drop MRI scan or click to browse — PNG, JPG</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# RUN INFERENCE ON UPLOAD
# ──────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ── Gatekeeper: reject non-MRI uploads ───────
    mri_valid, mri_confidence = is_brain_mri(image, gatekeeper_model, gatekeeper_device)
    if not mri_valid:
        st.error(
            "The uploaded image does not appear to be a valid Brain MRI scan. "
            "Please upload a valid scan."
        )
        st.stop()

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="label-display">INPUT MRI SCAN</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="label-display">⚡ DIAGNOSIS</p>', unsafe_allow_html=True)

        # Show scanning animation while inference runs
        progress = st.progress(0)
        status = st.empty()
        status.markdown('<p class="scan-text">ANALYZING NEURAL TISSUE</p>', unsafe_allow_html=True)

        # Animate progress bar (cosmetic — runs before inference completes)
        for i in range(60):
            time.sleep(0.01)
            progress.progress(i + 1)

        # Run real inference
        confidences, detected, overlay_np, elapsed = run_inference(image, model, grad_cam, device)

        # Finish progress bar
        for i in range(60, 100):
            progress.progress(i + 1)

        status.empty()
        progress.empty()

        # Determine if a tumor was detected
        is_tumor = detected != "notumor"

        # Status banner
        if is_tumor:
            st.markdown(
                f'<div class="status-detected">'
                f'⚠️ <div><p style="font-family:Orbitron;font-size:0.75rem;letter-spacing:0.1em;margin:0;color:#c8d6e5;">'
                f'ANOMALY DETECTED</p>'
                f'<p style="font-size:0.75rem;color:#5a6a7a;margin:0;">'
                f'{detected.capitalize()} — {confidences[detected]:.1f}% confidence</p></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-clear">'
                f'✅ <div><p style="font-family:Orbitron;font-size:0.75rem;letter-spacing:0.1em;margin:0;color:#c8d6e5;">'
                f'NO ANOMALY DETECTED</p>'
                f'<p style="font-size:0.75rem;color:#5a6a7a;margin:0;">'
                f'Healthy — {confidences[detected]:.1f}% confidence</p></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Confidence rings — show top 3 classes sorted by confidence
        sorted_classes = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
        ring_cols = st.columns(len(sorted_classes))
        for col, (label, conf) in zip(ring_cols, sorted_classes):
            with col:
                is_det = label == detected
                fig = create_confidence_ring(label, conf, is_det)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    f'<p style="text-align:center;font-size:0.7rem;color:#5a6a7a;font-family:Inter;">'
                    f'{label.capitalize()}</p>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<p class="footer-text">Model: ResNet50 + Grad-CAM — Inference: {elapsed:.2f}s</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # GRAD-CAM EXPLAINABILITY PANEL
    # ──────────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="label-display">🔬 EXPLAINABILITY — GRAD-CAM OVERLAY</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:#5a6a7a;font-family:Inter;">'
        'Highlights the regions the model focused on when making its prediction. '
        'Red/yellow areas indicate high importance; blue areas indicate low importance.</p>',
        unsafe_allow_html=True,
    )

    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        st.markdown(
            '<p style="text-align:center;font-size:0.7rem;color:#06b6d4;font-family:Orbitron;letter-spacing:0.1em;">'
            'ORIGINAL SCAN</p>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with cam_col2:
        st.markdown(
            '<p style="text-align:center;font-size:0.7rem;color:#06b6d4;font-family:Orbitron;letter-spacing:0.1em;">'
            'GRAD-CAM OVERLAY</p>',
            unsafe_allow_html=True,
        )
        # Convert overlay from [0,1] float to uint8 for display
        overlay_display = (overlay_np * 255).astype(np.uint8)
        st.image(overlay_display, use_container_width=True)

    with cam_col3:
        st.markdown(
            '<p style="text-align:center;font-size:0.7rem;color:#06b6d4;font-family:Orbitron;letter-spacing:0.1em;">'
            'CLASSIFICATION SUMMARY</p>',
            unsafe_allow_html=True,
        )
        # Show all class confidences as a sorted list
        for label, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
            bar_color = "#ef4444" if label == detected and is_tumor else "#22c55e" if label == detected else "#1e293b"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.4rem 0.6rem;margin:0.25rem 0;border-radius:0.5rem;'
                f'background:rgba(30,41,59,0.5);border-left:3px solid {bar_color};">'
                f'<span style="font-family:Inter;font-size:0.8rem;color:#c8d6e5;">{label.capitalize()}</span>'
                f'<span style="font-family:Orbitron;font-size:0.75rem;color:#c8d6e5;">{conf:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p style="text-align:center;font-size:0.65rem;color:#5a6a7a;font-family:Inter;margin-top:1rem;">'
        'Target Layer: backbone.layer4[-1] (ResNet50) — '
        'Grad-CAM (Selvaraju et al., 2017)</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
