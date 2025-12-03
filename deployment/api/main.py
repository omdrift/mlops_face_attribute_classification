"""
FastAPI application with Prometheus metrics integration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import torch
import numpy as np
from PIL import Image
import io
import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT_DIR)

from src.models.architecture import CustomMultiHeadCNN
from src.monitoring.prometheus_metrics import (
    get_metrics,
    get_metrics_content_type,
    track_inference_time,
    record_prediction,
    set_model_loaded,
    set_model_info,
    PrometheusMiddleware
)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Face Attribute Classification API",
    description="API for facial attribute classification with monitoring",
    version="1.0.0"
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Global model variable
model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    model_path = "models/best_model.pth"
    
    try:
        if os.path.exists(model_path):
            model = CustomMultiHeadCNN(n_color=5, n_length=3).to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Set Prometheus metrics
            set_model_loaded("v1.0.0", True)
            set_model_info("v1.0.0", "pytorch", model_path=model_path)
            
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Face Attribute Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Predict attributes from image",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )


@track_inference_time
def predict_image(image_tensor):
    """
    Predict attributes for image tensor
    
    Args:
        image_tensor: Preprocessed image tensor
    
    Returns:
        Dictionary of predictions
    """
    with torch.no_grad():
        outputs = model(image_tensor)
    
    predictions = {
        'beard': int(torch.argmax(outputs[0], dim=1).item()),
        'mustache': int(torch.argmax(outputs[1], dim=1).item()),
        'glasses': int(torch.argmax(outputs[2], dim=1).item()),
        'hair_color': int(torch.argmax(outputs[3], dim=1).item()),
        'hair_length': int(torch.argmax(outputs[4], dim=1).item()),
    }
    
    # Record predictions in Prometheus
    for attr, value in predictions.items():
        record_prediction(attr, value)
    
    return predictions


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict facial attributes from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary with predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize and normalize
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Predict
        predictions = predict_image(image_tensor)
        
        # Add human-readable labels
        result = {
            "predictions": predictions,
            "labels": {
                "beard": "Yes" if predictions['beard'] == 1 else "No",
                "mustache": "Yes" if predictions['mustache'] == 1 else "No",
                "glasses": "Yes" if predictions['glasses'] == 1 else "No",
                "hair_color": ["Black", "Blonde", "Brown", "Gray", "Red"][predictions['hair_color']],
                "hair_length": ["Short", "Medium", "Long"][predictions['hair_length']],
            }
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
