"""
FastAPI application for face attribute classification and search
"""
import os
import sys
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

# Add app to path
sys.path.insert(0, '/app')

from api.models import (
   AttributeValues, SearchRequest, SearchResponse, 
   ImageResult, PredictionResult, HealthResponse
)
from api.inference import initialize_predictor, get_predictor
from api.utils import filter_images_by_attributes, get_human_readable_labels

# Configuration
DATA_DIR = "/app/data"


@asynccontextmanager
async def lifespan(app: FastAPI):
   """Initialize predictor on startup"""
   initialize_predictor()
   yield


# Create FastAPI app
app = FastAPI(
   title="Face Attribute Classification API",
   description="API for searching images by facial attributes",
   version="1.0.0",
   lifespan=lifespan
)

# CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="/app/frontend/templates")
app.mount("/static", StaticFiles(directory="/app/frontend/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
   """Serve the frontend interface"""
   return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
   """Health check endpoint"""
   predictor = get_predictor()
   stats = predictor.get_stats()
   
   return HealthResponse(
       status="healthy" if stats["model_loaded"] else "unhealthy",
       model_loaded=stats["model_loaded"],
       images_scanned=stats["images_scanned"]
   )


@app.get("/api/attributes")
async def get_attributes():
   """Get list of available attributes and their values"""
   return AttributeValues()


@app.post("/api/search", response_model=SearchResponse)
async def search_images(search_request: SearchRequest):
   """
   Search images by facial attributes
   
   Supports multiple values per attribute (OR logic within each attribute, AND logic between attributes)
   """
   predictor = get_predictor()
   predictions_cache = predictor.get_predictions_cache()
   
   # Build search parameters from request
   search_params = {}
   if search_request.barbe is not None and len(search_request.barbe) > 0:
       search_params["barbe"] = search_request.barbe
   if search_request.moustache is not None and len(search_request.moustache) > 0:
       search_params["moustache"] = search_request.moustache
   if search_request.lunettes is not None and len(search_request.lunettes) > 0:
       search_params["lunettes"] = search_request.lunettes
   if search_request.taille_cheveux is not None and len(search_request.taille_cheveux) > 0:
       search_params["taille_cheveux"] = search_request.taille_cheveux
   if search_request.couleur_cheveux is not None and len(search_request.couleur_cheveux) > 0:
       search_params["couleur_cheveux"] = search_request.couleur_cheveux
   
   # Filter images
   matching_filenames = filter_images_by_attributes(predictions_cache, search_params)
   
   # Build response
   results = []
   for filename in matching_filenames:
       results.append(ImageResult(
           filename=filename,
           path=f"/api/images/{filename}",
           attributes=predictions_cache[filename]
       ))
   
   return SearchResponse(
       total=len(results),
       images=results
   )


@app.get("/api/images/{filename}")
async def get_image(filename: str):
   """Serve an image from the data directory"""
   image_path = os.path.join(DATA_DIR, filename)
   
   if not os.path.exists(image_path):
       raise HTTPException(status_code=404, detail="Image not found")
   
   return FileResponse(image_path)


@app.post("/api/predict", response_model=PredictionResult)
async def predict_attributes(file: UploadFile = File(...)):
   """
   Predict attributes for an uploaded image
   """
   if not file.content_type.startswith('image/'):
       raise HTTPException(status_code=400, detail="File must be an image")
   
   # Save uploaded file temporarily
   temp_path = f"/tmp/{file.filename}"
   try:
       contents = await file.read()
       with open(temp_path, 'wb') as f:
           f.write(contents)
       
       # Predict
       predictor = get_predictor()
       predictions = predictor.predict_single_image(temp_path)
       
       if predictions is None:
           raise HTTPException(status_code=500, detail="Error processing image")
       
       # Get human-readable labels
       labels = get_human_readable_labels(predictions)
       
       return PredictionResult(
           barbe=predictions["barbe"],
           moustache=predictions["moustache"],
           lunettes=predictions["lunettes"],
           taille_cheveux=predictions["taille_cheveux"],
           couleur_cheveux=predictions["couleur_cheveux"],
           labels=labels
       )
   finally:
       # Clean up
       if os.path.exists(temp_path):
           os.remove(temp_path)


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
