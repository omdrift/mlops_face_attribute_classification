# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Client                            │
└──────────────────┬────────────────────────────────┬─────────────┘
                   │                                │
                   │ HTTP                           │ HTTP
                   │                                │
        ┌──────────▼──────────┐          ┌─────────▼─────────┐
        │   Web Browser       │          │   API Client      │
        │  (Frontend UI)      │          │  (curl, scripts)  │
        └──────────┬──────────┘          └─────────┬─────────┘
                   │                                │
                   └────────────┬───────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Docker Container   │
                     │  (Port 8000)         │
                     │                      │
                     │  ┌────────────────┐  │
                     │  │  FastAPI App   │  │
                     │  │                │  │
                     │  │  Endpoints:    │  │
                     │  │  - GET /       │  │
                     │  │  - /api/*      │  │
                     │  └────────┬───────┘  │
                     │           │          │
                     │  ┌────────▼───────┐  │
                     │  │  Model Layer   │  │
                     │  │  (ResNet-18)   │  │
                     │  │                │  │
                     │  │  38 Attributes │  │
                     │  └────────┬───────┘  │
                     │           │          │
                     └───────────┼──────────┘
                                 │
                       Volume Mount
                                 │
                     ┌───────────▼──────────┐
                     │   Host File System   │
                     │   /data directory    │
                     │                      │
                     │  ├── images/         │
                     │  │   ├── img1.jpg    │
                     │  │   ├── img2.jpg    │
                     │  │   └── ...         │
                     │  │                   │
                     │  ├── annotations.csv │
                     │  │   (optional)      │
                     │  │                   │
                     │  └── model.pth       │
                     │      (optional)      │
                     └──────────────────────┘
```

## Data Flow

### 1. Image Search Flow (With Pre-computed Annotations)

```
User → Web UI → Select Attributes → API Request
                                         │
                                         ▼
                                   POST /api/search
                                         │
                                         ▼
                            Load annotations.csv
                                         │
                                         ▼
                            Filter by attributes
                                         │
                                         ▼
                            Calculate match scores
                                         │
                                         ▼
                            Return image list
                                         │
                                         ▼
                            Web UI displays results
```

### 2. Image Search Flow (On-the-fly Processing)

```
User → Web UI → Select Attributes → API Request
                                         │
                                         ▼
                                   POST /api/search
                                         │
                                         ▼
                            Scan images directory
                                         │
                                         ▼
                       For each image (up to MAX_IMAGES):
                                         │
                          ┌──────────────┴─────────────┐
                          ▼                            ▼
                    Load & Preprocess          Run Model Inference
                          │                            │
                          └──────────────┬─────────────┘
                                         ▼
                            Extract attribute scores
                                         │
                                         ▼
                            Filter by threshold
                                         │
                                         ▼
                            Calculate match scores
                                         │
                                         ▼
                            Return image list
                                         │
                                         ▼
                            Web UI displays results
```

## Component Details

### FastAPI Application (`src/api/app.py`)
- **Responsibilities:**
  - HTTP request handling
  - Route management
  - Response formatting
  - Image search logic
  - Database/CSV handling

- **Endpoints:**
  - `GET /` - Serve web interface
  - `GET /api/attributes` - List available attributes
  - `POST /api/search` - Search images by attributes
  - `GET /api/health` - Health check

### Model Layer (`src/model/face_attribute_model.py`)
- **Responsibilities:**
  - Model initialization
  - Image preprocessing
  - Attribute prediction
  - Probability calculation

- **Architecture:**
  - Base: ResNet-18 (pre-trained on ImageNet)
  - Modified: Final layer for 38-class multi-label classification
  - Output: Sigmoid activation for probabilities

### Web Frontend (`templates/index.html`)
- **Features:**
  - Attribute selection grid
  - Threshold adjustment
  - Real-time search
  - Image preview
  - Match score display
  - Responsive design

- **Security:**
  - DOM-based rendering (no innerHTML)
  - Input validation
  - XSS prevention

## Deployment Architecture

### Docker Container
- **Base Image:** Python 3.10 slim
- **Dependencies:** PyTorch, FastAPI, Uvicorn
- **Volume Mount:** `./data` → `/app/data`
- **Port Mapping:** `8000:8000`
- **Health Check:** HTTP check on `/api/health`

### Volume Structure
```
/app/data/
├── images/              # Required: Face images
│   ├── *.jpg
│   ├── *.png
│   └── ...
├── annotations.csv      # Optional: Pre-computed attributes
└── model.pth           # Optional: Trained model weights
```

## Performance Considerations

### With Pre-computed Annotations
- **Pros:** Fast search (CSV parsing only)
- **Cons:** Requires pre-processing
- **Use Case:** Production with large image sets

### On-the-fly Processing
- **Pros:** No pre-processing needed
- **Cons:** Slower, limited to MAX_IMAGES_TO_PROCESS
- **Use Case:** Development, small image sets

## Scalability Options

1. **Horizontal Scaling:** Run multiple container instances with load balancer
2. **GPU Support:** Add GPU runtime for faster inference
3. **Caching:** Add Redis for frequently accessed results
4. **Database:** Replace CSV with PostgreSQL/MongoDB for large datasets
5. **CDN:** Serve images through CDN for faster delivery
6. **Message Queue:** Add queue for async processing of large batches

## Security Features

1. **Input Validation:** Pydantic models for API requests
2. **Type Safety:** Type hints throughout codebase
3. **XSS Prevention:** DOM-based rendering in frontend
4. **Container Isolation:** Runs in isolated Docker container
5. **No Code Injection:** No eval() or exec() usage
6. **Health Checks:** Monitoring endpoint for container health
