"""
Model inference logic with caching
"""
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for model imports
sys.path.insert(0, '/app')

from src.models.architecture import CustomMultiHeadCNN
from api.utils import get_all_images_from_directory

# Configuration
MODEL_PATH = "/app/models/best_model.pth"
DATA_DIR = "/app/data"
CACHE_FILE = "/tmp/predictions_cache.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceAttributePredictor:
    """Face attribute prediction with caching"""
    
    def __init__(self):
        self.model = None
        self.predictions_cache = {}
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = CustomMultiHeadCNN(n_color=5, n_length=3).to(DEVICE)
            
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess image for model inference
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor or None if error
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize to 64x64
            img = img.resize((64, 64))
            
            # Convert to numpy array and normalize
            img_array = np.array(img).astype('float32') / 255.0
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
            return img_tensor.to(DEVICE)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_single_image(self, image_path: str) -> Optional[Dict[str, int]]:
        """
        Predict attributes for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with predicted attributes or None if error
        """
        if not self.model_loaded:
            logger.error("Model not loaded")
            return None
        
        img_tensor = self.preprocess_image(image_path)
        if img_tensor is None:
            return None
        
        try:
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            # Extract predictions
            predictions = {
                "barbe": int(torch.sigmoid(outputs['beard']) > 0.5),
                "moustache": int(torch.sigmoid(outputs['mustache']) > 0.5),
                "lunettes": int(torch.sigmoid(outputs['glasses']) > 0.5),
                "taille_cheveux": int(torch.argmax(outputs['hair_length'], dim=1)),
                "couleur_cheveux": int(torch.argmax(outputs['hair_color'], dim=1))
            }
            
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def scan_and_predict_all(self, force_refresh: bool = False):
        """
        Scan all images in data directory and predict their attributes
        
        Args:
            force_refresh: If True, re-predict all images even if cache exists
        """
        # Load cache if exists and not forcing refresh
        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    self.predictions_cache = json.load(f)
                logger.info(f"Loaded predictions cache with {len(self.predictions_cache)} images")
                return
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        # Get all images
        logger.info(f"Scanning images in {DATA_DIR}")
        images = get_all_images_from_directory(DATA_DIR)
        logger.info(f"Found {len(images)} images")
        
        # Predict attributes for each image
        for i, filename in enumerate(images):
            if i % 100 == 0:
                logger.info(f"Processing image {i+1}/{len(images)}")
            
            image_path = os.path.join(DATA_DIR, filename)
            predictions = self.predict_single_image(image_path)
            
            if predictions:
                self.predictions_cache[filename] = predictions
        
        # Save cache
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.predictions_cache, f)
            logger.info(f"Saved predictions cache with {len(self.predictions_cache)} images")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get_predictions_cache(self) -> Dict:
        """Get the predictions cache"""
        return self.predictions_cache
    
    def get_stats(self) -> Dict:
        """Get statistics about predictions"""
        return {
            "model_loaded": self.model_loaded,
            "images_scanned": len(self.predictions_cache)
        }


# Global predictor instance
predictor = FaceAttributePredictor()


def initialize_predictor():
    """Initialize the predictor (load model and scan images)"""
    predictor.load_model()
    predictor.scan_and_predict_all()


def get_predictor() -> FaceAttributePredictor:
    """Get the global predictor instance"""
    return predictor
