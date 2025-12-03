# Face Attribute Classification - MLOps Deployment

This project provides a web application and API for searching face images by attributes using a deep learning model.

## 🚀 Features

- **Face Attribute Classification**: Multi-label classification of 38 facial attributes
- **REST API**: FastAPI-based REST API for programmatic access
- **Web Interface**: User-friendly web interface for attribute-based image search
- **Docker Deployment**: Fully containerized application with Docker support
- **Volume Mounting**: Mount local image directory to `/app/data` in the container

## 📋 Attributes Supported

The model supports classification of the following facial attributes:
- Gender: Male
- Age: Young
- Appearance: Attractive, Pale_Skin, Chubby, Rosy_Cheeks
- Facial Features: Big_Lips, Big_Nose, Pointy_Nose, High_Cheekbones
- Hair: Wavy_Hair, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Bald, Bangs, Receding_Hairline, Straight_Hair
- Facial Hair: Goatee, Mustache, No_Beard, Sideburns, 5_o_Clock_Shadow
- Face Shape: Oval_Face, Double_Chin
- Eyes & Eyebrows: Arched_Eyebrows, Bushy_Eyebrows, Narrow_Eyes, Bags_Under_Eyes
- Accessories: Eyeglasses, Wearing_Hat, Wearing_Earrings, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie
- Other: Smiling, Heavy_Makeup

## 🏗️ Project Structure

```
mlops_face_attribute_classification/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── face_attribute_model.py   # ML model wrapper
│   └── api/
│       └── app.py                     # FastAPI application
├── templates/
│   └── index.html                     # Web frontend
├── static/                            # Static files
├── data/                              # Data directory (mounted as volume)
│   ├── images/                        # Image files
│   ├── annotations.csv                # (Optional) Pre-computed attributes
│   └── model.pth                      # (Optional) Trained model weights
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker Compose configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🐳 Docker Deployment

### Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)

### Option 1: Using Docker Compose (Recommended)

1. **Prepare your data directory**:
   ```bash
   mkdir -p data/images
   # Copy your face images to data/images/
   ```

2. **Build and run the container**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/health

### Option 2: Using Docker directly

1. **Build the Docker image**:
   ```bash
   docker build -t face-attribute-api .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     -e DATA_DIR=/app/data \
     -e MODEL_PATH=/app/data/model.pth \
     --name face-attribute-api \
     face-attribute-api
   ```

3. **Access the application**: http://localhost:8000

### Volume Mounting

The application expects the following directory structure in your mounted volume:

```
/app/data/
├── images/              # Your face image files (required)
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── annotations.csv      # (Optional) Pre-computed attributes for faster search
└── model.pth           # (Optional) Your trained model weights
```

**Important**: Make sure your local `data/images/` directory contains face images in supported formats (JPG, JPEG, PNG, BMP).

## 🖥️ Local Development

### Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data directory**:
   ```bash
   mkdir -p data/images
   # Copy your images to data/images/
   ```

4. **Run the application**:
   ```bash
   export DATA_DIR=./data  # On Windows: set DATA_DIR=./data
   uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the application**: http://localhost:8000

## 📡 API Endpoints

### GET /
- **Description**: Web interface for image search
- **Response**: HTML page

### GET /api/attributes
- **Description**: Get list of available attributes
- **Response**: 
  ```json
  {
    "attributes": ["Male", "Young", "Smiling", ...]
  }
  ```

### POST /api/search
- **Description**: Search for images by attributes
- **Request Body**:
  ```json
  {
    "attributes": ["Male", "Smiling"],
    "threshold": 0.5,
    "limit": 20
  }
  ```
- **Response**:
  ```json
  {
    "query": {
      "attributes": ["Male", "Smiling"],
      "threshold": 0.5,
      "limit": 20
    },
    "results": [
      {
        "filename": "image001.jpg",
        "path": "/images/image001.jpg",
        "attributes": {
          "Male": 0.95,
          "Smiling": 0.87
        },
        "match_score": 0.91
      }
    ],
    "count": 1
  }
  ```

### GET /api/health
- **Description**: Health check endpoint
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "data_dir": "/app/data",
    "images_dir_exists": true,
    "annotations_exists": false
  }
  ```

## 🎨 Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Select one or more facial attributes from the grid
3. Adjust the confidence threshold (0-1) using the slider
4. Set the maximum number of results
5. Click "🔍 Search Images"
6. View matching images with their attribute scores

## 🔧 Configuration

### Environment Variables

- `DATA_DIR`: Path to the data directory (default: `/app/data`)
- `MODEL_PATH`: Path to the trained model file (optional)

### Pre-computed Annotations

For faster search, you can provide a pre-computed CSV file with image attributes at `data/annotations.csv`:

```csv
filename,Male,Young,Smiling,Eyeglasses,...
image001.jpg,0.95,0.87,0.92,0.05,...
image002.jpg,0.12,0.95,0.76,0.88,...
```

The CSV should have:
- First column: `filename` (matching files in `data/images/`)
- Following columns: One column per attribute with probability values (0-1)

## 📝 Model Information

The application uses a ResNet-18 based model for multi-label facial attribute classification. By default, it uses ImageNet pre-trained weights. For better results:

1. Train a model on a face attribute dataset (e.g., CelebA)
2. Save the model weights as `data/model.pth`
3. The application will automatically load these weights on startup

## 🛠️ Troubleshooting

### No images found
- Ensure images are in `data/images/` directory
- Check that images are in supported formats (JPG, JPEG, PNG, BMP)
- Verify the volume mount is correct in Docker

### Model not loading
- Check that `MODEL_PATH` environment variable points to a valid `.pth` file
- Ensure the model architecture matches (ResNet-18 with modified final layer)

### Slow search
- Pre-compute attributes and provide `annotations.csv` for faster search
- Reduce the number of images or use a more powerful machine
- Consider using GPU support for faster inference

## 📦 Dependencies

- FastAPI: Web framework
- PyTorch: Deep learning framework
- Torchvision: Computer vision models
- Pillow: Image processing
- Uvicorn: ASGI server
- Pandas: Data manipulation
- Jinja2: Template engine

## 📄 License

This project is for educational and research purposes.

## 👥 Contributors

- MLOps Team

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
