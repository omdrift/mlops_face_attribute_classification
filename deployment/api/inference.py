import os
import sys
import json
import torch
import hashlib
import numpy as np
from PIL import Image
import cv2
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
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/best_model.pth")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp")
CACHE_FILE = os.path.join(CACHE_DIR, "predictions_cache.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Taille d'image utilisée pendant l'entraînement
IMAGE_SIZE = 64


def get_data_hash(data_dir: str) -> str:
    """Calcule un hash basé sur la liste des fichiers"""
    images = sorted(get_all_images_from_directory(data_dir))
    content = "|".join(images)
    return hashlib.md5(content.encode()).hexdigest()[:16]


class FaceAttributePredictor:
    """Face attribute prediction with persistent caching"""
    
    def __init__(self):
        self.model = None
        self.predictions_cache = {}
        self.model_loaded = False
        self.data_hash = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            logger.info(f"Device: {DEVICE}")
            
            self.model = CustomMultiHeadCNN(n_color=5, n_length=3).to(DEVICE)
            
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.model_loaded = True
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess image EXACTEMENT comme pendant l'entraînement
        """
        try:
            # === MÉTHODE 1: Avec OpenCV (comme l'entraînement) ===
            img = cv2.imread(image_path)
            if img is None:
                logger.debug(f"Cannot read image: {image_path}")
                return None
            
            # Convertir BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Crop sur le contenu (enlever fond blanc) - IMPORTANT!
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
            coords = cv2.findNonZero(thresh)
            
            if coords is not None and len(coords) > 0:
                x, y, w, h = cv2.boundingRect(coords)
                # Vérifier que le crop est valide
                if w > 10 and h > 10:
                    img = img[y:y+h, x:x+w]
            
            # Resize à la taille d'entraînement
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Normaliser [0, 255] → [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convertir en tensor (H, W, C) → (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            return img_tensor.to(DEVICE)
            
        except Exception as e:
            logger.debug(f"Error preprocessing {image_path}: {e}")
            return None
    
    def preprocess_image_pil(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Alternative: Preprocess avec PIL (si OpenCV pose problème)
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Crop sur le contenu (enlever fond blanc)
            gray = np.mean(img_array, axis=2)
            mask = gray < 250  # Pixels non-blancs
            
            if mask.any():
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # Vérifier que le crop est valide
                if (y_max - y_min) > 10 and (x_max - x_min) > 10:
                    img_array = img_array[y_min:y_max+1, x_min:x_max+1]
            
            # Resize
            img = Image.fromarray(img_array)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            
            # Normaliser
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Tensor
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
            return img_tensor.to(DEVICE)
            
        except Exception as e:
            logger.debug(f"Error preprocessing {image_path}: {e}")
            return None
    
    def predict_single_image(self, image_path: str) -> Optional[Dict[str, int]]:
        """Predict attributes for a single image"""
        if not self.model_loaded:
            logger.error("Model not loaded")
            return None
        
        # Utiliser le prétraitement OpenCV (comme l'entraînement)
        img_tensor = self.preprocess_image(image_path)
        
        # Fallback sur PIL si OpenCV échoue
        if img_tensor is None:
            img_tensor = self.preprocess_image_pil(image_path)
        
        if img_tensor is None:
            return None
        
        try:
            with torch.no_grad():
                # Le modèle retourne un tuple
                out_beard, out_mustache, out_glasses, out_color, out_length = self.model(img_tensor)
            
            predictions = {
                "barbe": int(torch.sigmoid(out_beard).item() > 0.5),
                "moustache": int(torch.sigmoid(out_mustache).item() > 0.5),
                "lunettes": int(torch.sigmoid(out_glasses).item() > 0.5),
                "taille_cheveux": int(torch.argmax(out_length, dim=1).item()),
                "couleur_cheveux": int(torch.argmax(out_color, dim=1).item())
            }
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def is_cache_valid(self) -> bool:
        """Vérifie si le cache est valide"""
        if not os.path.exists(CACHE_FILE):
            return False
        
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            cached_hash = cache_data.get("data_hash", "")
            current_hash = get_data_hash(DATA_DIR)
            
            if cached_hash == current_hash:
                logger.info(f"✅ Cache is valid (hash: {current_hash})")
                return True
            else:
                logger.info(f"⚠️ Cache outdated (cached: {cached_hash}, current: {current_hash})")
                return False
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    def load_cache(self) -> bool:
        """Charge le cache depuis le fichier"""
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            self.predictions_cache = cache_data.get("predictions", {})
            self.data_hash = cache_data.get("data_hash", "")
            
            logger.info(f"✅ Loaded {len(self.predictions_cache)} predictions from cache")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False
    
    def save_cache(self):
        """Sauvegarde le cache"""
        try:
            cache_data = {
                "data_hash": self.data_hash,
                "predictions": self.predictions_cache,
                "total_images": len(self.predictions_cache)
            }
            
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"✅ Saved {len(self.predictions_cache)} predictions to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def scan_and_predict_all(self, force_refresh: bool = False):
        """Scan all images and predict their attributes"""
        
        if not force_refresh and self.is_cache_valid():
            self.load_cache()
            return
        
        self.data_hash = get_data_hash(DATA_DIR)
        
        logger.info(f"📂 Scanning images in {DATA_DIR}")
        images = get_all_images_from_directory(DATA_DIR)
        total = len(images)
        logger.info(f"📊 Found {total} images")
        
        if total == 0:
            logger.warning("⚠️ No images found!")
            return
        
        self.predictions_cache = {}
        
        for i, filename in enumerate(images):
            if i % 500 == 0:
                progress = (i / total) * 100
                logger.info(f"🔄 Processing {i+1}/{total} ({progress:.1f}%)")
            
            image_path = os.path.join(DATA_DIR, filename)
            predictions = self.predict_single_image(image_path)
            
            if predictions:
                self.predictions_cache[filename] = predictions
        
        logger.info(f"✅ Processed {len(self.predictions_cache)}/{total} images successfully")
        self.save_cache()
    
    def get_predictions_cache(self) -> Dict:
        return self.predictions_cache
    
    def get_stats(self) -> Dict:
        stats = {
            "model_loaded": self.model_loaded,
            "images_scanned": len(self.predictions_cache),
            "cache_file": CACHE_FILE,
            "data_hash": self.data_hash
        }
        
        if self.predictions_cache:
            attr_stats = {
                "barbe": {0: 0, 1: 0},
                "moustache": {0: 0, 1: 0},
                "lunettes": {0: 0, 1: 0},
                "taille_cheveux": {0: 0, 1: 0, 2: 0},
                "couleur_cheveux": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            }
            
            for preds in self.predictions_cache.values():
                for attr, val in preds.items():
                    if attr in attr_stats and val in attr_stats[attr]:
                        attr_stats[attr][val] += 1
            
            stats["attribute_distribution"] = attr_stats
        
        return stats


# Global predictor instance
predictor = FaceAttributePredictor()


def initialize_predictor():
    """Initialize the predictor"""
    predictor.load_model()
    predictor.scan_and_predict_all()


def get_predictor() -> FaceAttributePredictor:
    return predictor