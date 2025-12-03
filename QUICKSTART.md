# Quick Start Guide

## 🚀 Getting Started with Face Attribute Classification API

This guide will help you deploy the Face Attribute Classification API quickly.

### Prerequisites

- Docker and Docker Compose installed
- Face images in a local directory
- (Optional) Trained model weights file

### Step 1: Prepare Your Data

Create the following directory structure:

```
mlops_face_attribute_classification/
├── data/
│   ├── images/           # Place your face images here
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── annotations.csv   # (Optional) Pre-computed attributes
│   └── model.pth         # (Optional) Your trained model
```

**Important:** 
- All your face images should be placed in `data/images/` directory
- Supported formats: JPG, JPEG, PNG, BMP
- The Docker container will mount this directory to `/app/data`

### Step 2: Deploy with Docker Compose

From the project root directory:

```bash
# Build and start the container
docker compose up --build

# Or run in detached mode
docker compose up --build -d
```

This will:
- Build the Docker image with all dependencies
- Mount your local `./data` directory to `/app/data` in the container
- Start the API server on port 8000

### Step 3: Access the Application

Once the container is running:

1. **Web Interface:** Open http://localhost:8000 in your browser
2. **API Documentation:** Visit http://localhost:8000/docs for interactive API docs
3. **Health Check:** Check http://localhost:8000/api/health to verify the API is running

### Step 4: Search for Images

#### Using the Web Interface:

1. Go to http://localhost:8000
2. Select one or more facial attributes (e.g., "Male", "Smiling")
3. Adjust the confidence threshold (0-1)
4. Click "🔍 Search Images"
5. View the matching images with their attribute scores

#### Using the API:

```bash
# Search for male and smiling faces
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": ["Male", "Smiling"],
    "threshold": 0.7,
    "limit": 10
  }'
```

### Alternative: Local Development

If you prefer to run without Docker:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export DATA_DIR=./data
export MODEL_PATH=./data/model.pth  # Optional

# 4. Run the server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Optional: Pre-compute Attributes

For faster search, you can pre-compute attributes for all images:

```bash
# Install dependencies first
pip install -r requirements.txt

# Generate annotations
python scripts/generate_annotations.py \
  --images-dir data/images \
  --output data/annotations.csv \
  --model-path data/model.pth  # Optional
```

This creates `data/annotations.csv` with pre-computed attributes for all images.

### Stopping the Application

```bash
# If running with Docker Compose
docker compose down

# To also remove volumes
docker compose down -v
```

### Troubleshooting

#### No images found
- Ensure images are in `data/images/` directory
- Check image formats (JPG, JPEG, PNG, BMP only)
- Verify Docker volume mount: `docker compose config` should show the volume mapping

#### Slow search
- Pre-compute attributes using `scripts/generate_annotations.py`
- Use a more powerful machine or enable GPU support
- Reduce the number of images or use pagination

#### Model not loading
- Check that `MODEL_PATH` points to a valid `.pth` file
- Ensure the model architecture matches (ResNet-18 based)
- The API will use base model if custom weights can't be loaded

### Next Steps

- Train your own model on face attribute datasets (e.g., CelebA)
- Customize the attributes list in `src/model/face_attribute_model.py`
- Add authentication and rate limiting for production use
- Deploy to cloud platforms (AWS, GCP, Azure)

### Support

For issues or questions:
1. Check the main README.md for detailed documentation
2. Review the API docs at http://localhost:8000/docs
3. Open an issue on GitHub
