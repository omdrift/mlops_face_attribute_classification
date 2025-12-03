"""
Prometheus metrics integration for FastAPI
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)
from typing import Callable
import time
from functools import wraps


# Create a custom registry
registry = CollectorRegistry()

# Define metrics

# Counters
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

model_predictions_total = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['attribute', 'value'],
    registry=registry
)

drift_alerts_total = Counter(
    'drift_alerts_total',
    'Total number of drift alerts',
    ['attribute', 'severity'],
    registry=registry
)

# Histograms
api_request_latency_seconds = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint'],
    registry=registry,
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

model_inference_latency_seconds = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency in seconds',
    registry=registry,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

batch_processing_time_seconds = Histogram(
    'batch_processing_time_seconds',
    'Batch processing time in seconds',
    registry=registry,
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

# Gauges
model_loaded = Gauge(
    'model_loaded',
    'Model loaded status (1=loaded, 0=not loaded)',
    ['version'],
    registry=registry
)

images_in_cache_total = Gauge(
    'images_in_cache_total',
    'Number of images in cache',
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy by attribute',
    ['attribute'],
    registry=registry
)

drift_score = Gauge(
    'drift_score',
    'Drift score by attribute',
    ['attribute'],
    registry=registry
)

# Info
model_info = Info(
    'model_info',
    'Information about the loaded model',
    registry=registry
)


def track_request_metrics(endpoint: str):
    """
    Decorator to track request metrics
    
    Usage:
        @track_request_metrics('/predict')
        async def predict(request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                api_request_latency_seconds.labels(endpoint=endpoint).observe(duration)
                api_requests_total.labels(
                    method='POST',
                    endpoint=endpoint,
                    status=str(status)
                ).inc()
        
        return wrapper
    return decorator


def track_inference_time(func: Callable) -> Callable:
    """
    Decorator to track model inference time
    
    Usage:
        @track_inference_time
        def predict(model, inputs):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        model_inference_latency_seconds.observe(duration)
        
        return result
    
    return wrapper


def record_prediction(attribute: str, value: int):
    """
    Record a prediction for an attribute
    
    Args:
        attribute: Attribute name (beard, mustache, etc.)
        value: Predicted value
    """
    model_predictions_total.labels(
        attribute=attribute,
        value=str(value)
    ).inc()


def record_drift_alert(attribute: str, severity: str = 'medium'):
    """
    Record a drift alert
    
    Args:
        attribute: Attribute with drift
        severity: Alert severity (low, medium, high)
    """
    drift_alerts_total.labels(
        attribute=attribute,
        severity=severity
    ).inc()


def update_model_accuracy(accuracies: dict):
    """
    Update model accuracy metrics
    
    Args:
        accuracies: Dictionary of attribute -> accuracy
    """
    for attribute, accuracy in accuracies.items():
        model_accuracy.labels(attribute=attribute).set(accuracy)


def update_drift_scores(drift_scores: dict):
    """
    Update drift score metrics
    
    Args:
        drift_scores: Dictionary of attribute -> drift score
    """
    for attribute, score in drift_scores.items():
        drift_score.labels(attribute=attribute).set(score)


def set_model_loaded(version: str, loaded: bool = True):
    """
    Set model loaded status
    
    Args:
        version: Model version
        loaded: Whether model is loaded
    """
    model_loaded.labels(version=version).set(1 if loaded else 0)


def set_model_info(version: str, framework: str = 'pytorch', **kwargs):
    """
    Set model information
    
    Args:
        version: Model version
        framework: ML framework used
        **kwargs: Additional info
    """
    info = {
        'version': version,
        'framework': framework,
        **kwargs
    }
    model_info.info(info)


def get_metrics():
    """
    Get current metrics in Prometheus format
    
    Returns:
        Metrics as bytes
    """
    return generate_latest(registry)


def get_metrics_content_type():
    """
    Get content type for metrics response
    
    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST


# FastAPI middleware
class PrometheusMiddleware:
    """
    Middleware to automatically track all requests
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope.get('method', 'UNKNOWN')
        path = scope.get('path', '/')
        
        # Status code tracking
        status_code = 200
        
        async def send_wrapper(message):
            nonlocal status_code
            if message['type'] == 'http.response.start':
                status_code = message.get('status', 200)
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            api_request_latency_seconds.labels(endpoint=path).observe(duration)
            api_requests_total.labels(
                method=method,
                endpoint=path,
                status=str(status_code)
            ).inc()


if __name__ == '__main__':
    # Example usage
    print("Prometheus Metrics - Example Usage")
    
    # Simulate some metrics
    api_requests_total.labels(method='POST', endpoint='/predict', status='200').inc()
    api_request_latency_seconds.labels(endpoint='/predict').observe(0.045)
    model_inference_latency_seconds.observe(0.023)
    
    record_prediction('beard', 1)
    record_prediction('mustache', 0)
    
    update_model_accuracy({
        'beard': 0.92,
        'mustache': 0.89,
        'glasses': 0.95,
    })
    
    set_model_loaded('v1.0.0', True)
    
    # Print metrics
    print("\nMetrics:")
    print(get_metrics().decode('utf-8'))
