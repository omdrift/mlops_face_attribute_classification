from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.face_attribute_model import FaceAttributeModel

app = FastAPI(title="Face Attribute Classification API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global model instance
model = None
image_database = None

# Data directory (will be mounted in Docker)
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
ANNOTATIONS_PATH = os.path.join(DATA_DIR, 'annotations.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')


class AttributeQuery(BaseModel):
    """Request model for attribute search."""
    attributes: List[str]
    threshold: float = 0.5
    limit: int = 20


class ImageResult(BaseModel):
    """Response model for image results."""
    filename: str
    path: str
    attributes: Dict[str, float]
    match_score: float


def load_image_database():
    """Load or create image database with attributes."""
    global image_database
    
    # Check if annotations file exists
    if os.path.exists(ANNOTATIONS_PATH):
        print(f"Loading annotations from {ANNOTATIONS_PATH}")
        image_database = pd.read_csv(ANNOTATIONS_PATH)
    else:
        print("No annotations file found. Will process images on-the-fly.")
        image_database = None
    
    return image_database


def get_image_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image_attributes(image_path: str) -> Dict[str, float]:
    """Predict attributes for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = get_image_transform()
        image_tensor = transform(image).unsqueeze(0)
        
        attributes = model.predict_attributes(image_tensor)
        return attributes
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {}


def search_images_by_attributes(
    query_attributes: List[str],
    threshold: float = 0.5,
    limit: int = 20
) -> List[ImageResult]:
    """
    Search for images matching the specified attributes.
    
    Args:
        query_attributes: List of attribute names to search for
        threshold: Minimum probability threshold for matching
        limit: Maximum number of results to return
        
    Returns:
        List of matching images with their attributes
    """
    results = []
    
    # Check if images directory exists
    if not os.path.exists(IMAGES_DIR):
        print(f"Warning: Images directory not found at {IMAGES_DIR}")
        return results
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to search")
    
    # If we have a precomputed database, use it
    if image_database is not None:
        for idx, row in image_database.iterrows():
            if len(results) >= limit:
                break
            
            filename = row.get('filename', row.get('image_id', ''))
            image_path = os.path.join(IMAGES_DIR, filename)
            
            if not os.path.exists(image_path):
                continue
            
            # Calculate match score
            match_count = 0
            total_score = 0.0
            attributes = {}
            
            for attr in query_attributes:
                if attr in row:
                    prob = float(row[attr])
                    attributes[attr] = prob
                    if prob >= threshold:
                        match_count += 1
                        total_score += prob
            
            # Only include if all query attributes match
            if match_count == len(query_attributes):
                match_score = total_score / len(query_attributes)
                results.append(ImageResult(
                    filename=filename,
                    path=f"/images/{filename}",
                    attributes=attributes,
                    match_score=match_score
                ))
        
        # Sort by match score
        results.sort(key=lambda x: x.match_score, reverse=True)
    
    else:
        # Process images on-the-fly
        for image_path in image_files[:100]:  # Limit to avoid timeout
            if len(results) >= limit:
                break
            
            attributes = predict_image_attributes(image_path)
            
            if not attributes:
                continue
            
            # Calculate match score
            match_count = 0
            total_score = 0.0
            matching_attrs = {}
            
            for attr in query_attributes:
                if attr in attributes:
                    prob = attributes[attr]
                    matching_attrs[attr] = prob
                    if prob >= threshold:
                        match_count += 1
                        total_score += prob
            
            # Only include if all query attributes match
            if match_count == len(query_attributes):
                filename = os.path.basename(image_path)
                match_score = total_score / len(query_attributes)
                results.append(ImageResult(
                    filename=filename,
                    path=f"/images/{filename}",
                    attributes=matching_attrs,
                    match_score=match_score
                ))
        
        # Sort by match score
        results.sort(key=lambda x: x.match_score, reverse=True)
    
    return results[:limit]


@app.on_event("startup")
async def startup_event():
    """Initialize model and database on startup."""
    global model
    
    print("Starting Face Attribute Classification API...")
    
    # Initialize model
    model_path = os.getenv('MODEL_PATH', None)
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = FaceAttributeModel(model_path=model_path)
    else:
        print("No model path provided or file not found. Using base model.")
        model = FaceAttributeModel()
    
    # Load image database
    load_image_database()
    
    print("API started successfully!")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    attributes = model.get_attribute_list() if model else []
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "attributes": attributes}
    )


@app.get("/api/attributes")
async def get_attributes() -> Dict[str, List[str]]:
    """Get list of available attributes."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return {
        "attributes": model.get_attribute_list()
    }


@app.post("/api/search")
async def search_images(query: AttributeQuery) -> Dict[str, any]:
    """
    Search for images with specified attributes.
    
    Args:
        query: Query parameters including attributes, threshold, and limit
        
    Returns:
        Dictionary with search results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Validate attributes
    valid_attributes = model.get_attribute_list()
    invalid_attrs = [attr for attr in query.attributes if attr not in valid_attributes]
    
    if invalid_attrs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid attributes: {invalid_attrs}. Valid attributes: {valid_attributes}"
        )
    
    # Search for matching images
    results = search_images_by_attributes(
        query_attributes=query.attributes,
        threshold=query.threshold,
        limit=query.limit
    )
    
    return {
        "query": {
            "attributes": query.attributes,
            "threshold": query.threshold,
            "limit": query.limit
        },
        "results": [r.dict() for r in results],
        "count": len(results)
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_dir": DATA_DIR,
        "images_dir_exists": os.path.exists(IMAGES_DIR),
        "annotations_exists": os.path.exists(ANNOTATIONS_PATH)
    }


# Serve images from the data directory
if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
