"""
Pydantic models for API request/response validation
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AttributeValues(BaseModel):
    """Available values for each attribute"""
    barbe: List[Dict[str, Any]] = Field(
        default=[
            {"value": 0, "label": "Non"},
            {"value": 1, "label": "Oui"}
        ]
    )
    moustache: List[Dict[str, Any]] = Field(
        default=[
            {"value": 0, "label": "Non"},
            {"value": 1, "label": "Oui"}
        ]
    )
    lunettes: List[Dict[str, Any]] = Field(
        default=[
            {"value": 0, "label": "Non"},
            {"value": 1, "label": "Oui"}
        ]
    )
    taille_cheveux: List[Dict[str, Any]] = Field(
        default=[
            {"value": 0, "label": "Chauve"},
            {"value": 1, "label": "Court"},
            {"value": 2, "label": "Long"}
        ]
    )
    couleur_cheveux: List[Dict[str, Any]] = Field(
        default=[
            {"value": 0, "label": "Blond"},
            {"value": 1, "label": "Châtain"},
            {"value": 2, "label": "Roux"},
            {"value": 3, "label": "Brun"},
            {"value": 4, "label": "Gris/Blanc"}
        ]
    )


class SearchRequest(BaseModel):
    """Request model for image search by attributes"""
    barbe: Optional[List[int]] = Field(default=None, description="Beard values (0: non, 1: oui)")
    moustache: Optional[List[int]] = Field(default=None, description="Mustache values (0: non, 1: oui)")
    lunettes: Optional[List[int]] = Field(default=None, description="Glasses values (0: non, 1: oui)")
    taille_cheveux: Optional[List[int]] = Field(default=None, description="Hair length values (0: chauve, 1: court, 2: long)")
    couleur_cheveux: Optional[List[int]] = Field(default=None, description="Hair color values (0: blond, 1: châtain, 2: roux, 3: brun, 4: gris/blanc)")


class ImageResult(BaseModel):
    """Single image result with its attributes"""
    filename: str
    path: str
    attributes: Dict[str, int]


class SearchResponse(BaseModel):
    """Response model for image search"""
    total: int
    images: List[ImageResult]


class PredictionResult(BaseModel):
    """Prediction result for a single uploaded image"""
    barbe: int
    moustache: int
    lunettes: int
    taille_cheveux: int
    couleur_cheveux: int
    labels: Dict[str, str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    images_scanned: int
