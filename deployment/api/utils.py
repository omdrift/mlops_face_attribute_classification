import os
from typing import Dict


# Label mappings for human-readable output
ATTRIBUTE_LABELS = {
   "barbe": {0: "Non", 1: "Oui"},
   "moustache": {0: "Non", 1: "Oui"},
   "lunettes": {0: "Non", 1: "Oui"},
   "taille_cheveux": {0: "Chauve", 1: "Court", 2: "Long"},
   "couleur_cheveux": {0: "Blond", 1: "Châtain", 2: "Roux", 3: "Brun", 4: "Gris/Blanc"}
}


def get_human_readable_labels(predictions: Dict[str, int]) -> Dict[str, str]:
   """
   Convert numerical predictions to human-readable labels
   
   Args:
       predictions: Dictionary with attribute names and their predicted values
       
   Returns:
       Dictionary with attribute names and their human-readable labels
   """
   labels = {}
   for attr, value in predictions.items():
       if attr in ATTRIBUTE_LABELS:
           labels[attr] = ATTRIBUTE_LABELS[attr].get(value, "Inconnu")
   return labels


def get_all_images_from_directory(directory: str) -> list:
   """
   Get all image files from a directory
   
   Args:
       directory: Path to the directory containing images
       
   Returns:
       List of image filenames
   """
   if not os.path.exists(directory):
       return []
   
   valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
   images = []
   
   for filename in os.listdir(directory):
       ext = os.path.splitext(filename)[1].lower()
       if ext in valid_extensions:
           images.append(filename)
   
   return sorted(images)


def filter_images_by_attributes(predictions_cache: Dict, search_params: Dict) -> list:
   """
   Filter images based on search parameters
   
   Args:
       predictions_cache: Dictionary mapping filenames to their predicted attributes
       search_params: Dictionary with attribute filters (can have multiple values per attribute)
       
   Returns:
       List of matching filenames
   """
   matching_images = []
   
   for filename, attrs in predictions_cache.items():
       # Check if image matches all specified filters
       match = True
       
       for attr_name, allowed_values in search_params.items():
           if allowed_values is None or len(allowed_values) == 0:
               continue
               
           # Get the predicted value for this attribute
           predicted_value = attrs.get(attr_name)
           
           # Check if predicted value is in the allowed values (OR logic)
           if predicted_value not in allowed_values:
               match = False
               break
       
       if match:
           matching_images.append(filename)
   
   return matching_images