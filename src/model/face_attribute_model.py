import torch
import torch.nn as nn
from torchvision import models
from typing import List, Dict
import os


class FaceAttributeModel:
    """Face attribute classification model wrapper."""
    
    # Common face attributes for CelebA-like datasets
    ATTRIBUTES = [
        'Male', 'Young', 'Smiling', 'Eyeglasses', 'Attractive',
        'Wavy_Hair', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
        'Bald', 'Bangs', 'Receding_Hairline', 'Straight_Hair',
        'Wearing_Hat', 'Wearing_Earrings', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Heavy_Makeup', 'Pale_Skin', 'Rosy_Cheeks',
        'Big_Lips', 'Big_Nose', 'Pointy_Nose', 'High_Cheekbones',
        'Arched_Eyebrows', 'Bushy_Eyebrows', 'Narrow_Eyes',
        'Bags_Under_Eyes', 'Double_Chin', 'Goatee', 'Mustache',
        'No_Beard', 'Sideburns', '5_o_Clock_Shadow', 'Chubby', 'Oval_Face'
    ]
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the face attribute model.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create a simple ResNet-based model for multi-label classification
        self.num_attributes = len(self.ATTRIBUTES)
        self.model = models.resnet18(pretrained=True)
        
        # Replace the final layer for multi-label classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_attributes)
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Using model with pretrained ImageNet weights only")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict_attributes(self, image_tensor: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Predict attributes for a given image.
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W)
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary mapping attribute names to probabilities
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
        return {
            attr: float(prob)
            for attr, prob in zip(self.ATTRIBUTES, probabilities)
        }
    
    def get_attribute_list(self) -> List[str]:
        """Get the list of supported attributes."""
        return self.ATTRIBUTES.copy()
