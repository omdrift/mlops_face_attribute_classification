"""
Prometheus Metrics for ML Model API

This module defines and manages Prometheus metrics for monitoring the face
attribute classification API and model performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable, Any


# ============================================================================
# API Metrics
# ============================================================================

# Request metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

api_request_latency_seconds = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint', 'method'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ============================================================================
# Model Inference Metrics
# ============================================================================

model_predictions_total = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['attribute', 'predicted_value']
)

model_inference_latency_seconds = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency in seconds',
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
)

model_confidence_score = Histogram(
    'model_confidence_score',
    'Distribution of model confidence scores',
    ['attribute'],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
)

# ============================================================================
# Image Processing Metrics
# ============================================================================

images_processed_total = Counter(
    'images_processed_total',
    'Total number of images processed',
    ['status']  # success, error
)

# ============================================================================
# System Metrics
# ============================================================================

model_loaded = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

cache_size = Gauge(
    'cache_size',
    'Size of the prediction cache'
)

# ============================================================================
# Drift Metrics
# ============================================================================

drift_score = Gauge(
    'drift_score',
    'Current drift score for each attribute',
    ['attribute']
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy for each attribute',
    ['attribute']
)

# ============================================================================
# Model Information
# ============================================================================

model_info = Info(
    'model_info',
    'Information about the loaded model'
)


# ============================================================================
# Decorators for automatic metric tracking
# ============================================================================

def track_inference_time(func: Callable) -> Callable:
    """
    Decorator to track model inference time
    
    Usage:
        @track_inference_time
        def predict(image):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            model_inference_latency_seconds.observe(duration)
    
    return wrapper


def track_api_call(endpoint: str):
    """
    Decorator to track API calls
    
    Usage:
        @track_api_call('/predict')
        def predict_endpoint():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                api_request_latency_seconds.labels(
                    endpoint=endpoint,
                    method='POST'
                ).observe(duration)
                api_requests_total.labels(
                    endpoint=endpoint,
                    method='POST',
                    status=status
                ).inc()
        
        return wrapper
    
    return decorator


# ============================================================================
# Utility functions
# ============================================================================

def record_prediction(
    attribute: str,
    predicted_value: Any,
    confidence: float
):
    """
    Record a prediction with its metrics
    
    Args:
        attribute: Name of the attribute (e.g., 'beard', 'hair_color')
        predicted_value: Predicted value
        confidence: Confidence score (0-1)
    """
    # Increment prediction counter
    model_predictions_total.labels(
        attribute=attribute,
        predicted_value=str(predicted_value)
    ).inc()
    
    # Record confidence score
    model_confidence_score.labels(attribute=attribute).observe(confidence)


def update_drift_metrics(drift_scores: dict):
    """
    Update drift metrics for all attributes
    
    Args:
        drift_scores: Dictionary mapping attribute names to drift scores
    """
    for attribute, score in drift_scores.items():
        drift_score.labels(attribute=attribute).set(score)


def update_accuracy_metrics(accuracies: dict):
    """
    Update accuracy metrics for all attributes
    
    Args:
        accuracies: Dictionary mapping attribute names to accuracy scores
    """
    for attribute, accuracy in accuracies.items():
        model_accuracy.labels(attribute=attribute).set(accuracy)


def set_model_info(
    model_name: str,
    model_version: str,
    architecture: str,
    trained_date: str
):
    """
    Set model information
    
    Args:
        model_name: Name of the model
        model_version: Version identifier
        architecture: Model architecture name
        trained_date: Training date (ISO format)
    """
    model_info.info({
        'model_name': model_name,
        'version': model_version,
        'architecture': architecture,
        'trained_date': trained_date
    })


def set_model_loaded_status(loaded: bool):
    """
    Set whether the model is loaded
    
    Args:
        loaded: True if model is loaded, False otherwise
    """
    model_loaded.set(1 if loaded else 0)


def update_cache_size(size: int):
    """
    Update the cache size metric
    
    Args:
        size: Number of items in cache
    """
    cache_size.set(size)


# ============================================================================
# Example usage in FastAPI
# ============================================================================

"""
Example integration in FastAPI app:

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from src.monitoring.prometheus_metrics import (
    track_api_call, 
    record_prediction,
    set_model_loaded_status
)

app = FastAPI()

# Mount prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.on_event("startup")
async def startup_event():
    # Load model and set status
    load_model()
    set_model_loaded_status(True)

@app.post("/predict")
@track_api_call("/predict")
async def predict(image: UploadFile):
    # Perform prediction
    predictions = model.predict(image)
    
    # Record metrics
    for attr, (value, confidence) in predictions.items():
        record_prediction(attr, value, confidence)
    
    return predictions
"""
