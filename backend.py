import os
import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F_torch
from PIL import Image
import numpy as np

# Import our modular ML pipeline
from phase2_model_architecture import build_model
from phase5_gradcam import GradCAM, denormalize_image, overlay_heatmap
from phase1_data_pipeline import CLASS_NAMES, IMG_SIZE, get_device
from torchvision import transforms

app = FastAPI(title="Brain Tumor Detection API")

# Allow the Vite React dev server to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching the model
device = None
model = None
grad_cam = None

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.on_event("startup")
def load_model():
    global device, model, grad_cam
    print("Loading model weights...")
    device = get_device()
    model = build_model(device)
    
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = model.backbone.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    print("Model and Grad-CAM loaded successfully!")

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Forward pass for confidences
        with torch.no_grad():
            logits = model(input_tensor)
        probabilities = F_torch.softmax(logits, dim=1).squeeze()
        
        predicted_idx = probabilities.argmax().item()
        predicted_class = CLASS_NAMES[predicted_idx]
        
        confidences = {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        # Grad-CAM (Requires grad tracking)
        input_for_cam = preprocess(image).unsqueeze(0).to(device)
        heatmap, _ = grad_cam.generate(input_for_cam, target_class=predicted_idx)
        
        original_np = denormalize_image(input_for_cam.squeeze(0))
        overlay_np, _ = overlay_heatmap(original_np, heatmap)
        
        # Convert overlay numpy array (RGB) to base64 JPEG
        overlay_pil = Image.fromarray((overlay_np * 255).astype(np.uint8))
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="JPEG")
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidences": confidences,
            "overlay_image": f"data:image/jpeg;base64,{overlay_b64}"
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
