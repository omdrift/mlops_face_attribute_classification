#!/bin/bash
# Quick start script for building and testing the deployment

set -e  # Exit on error

echo "======================================"
echo "Face Attribute API - Quick Start"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "deployment/Dockerfile" ]; then
    echo "Error: Must be run from the project root directory"
    exit 1
fi

# Check if model exists
if [ ! -f "models/best_model.pth" ]; then
    echo "Warning: Model file models/best_model.pth not found"
    echo "You need to train the model first or place a trained model in models/"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

echo ""
echo "Step 1: Building Docker image..."
docker build -t face-attribute-api -f deployment/Dockerfile .

echo ""
echo "Step 2: Creating test images directory..."
mkdir -p deployment/images

echo ""
echo "Step 3: Checking if container is already running..."
if [ "$(docker ps -q -f name=face-attribute-api)" ]; then
    echo "Stopping existing container..."
    docker stop face-attribute-api
    docker rm face-attribute-api
fi

echo ""
echo "Step 4: Starting container..."
docker run -d \
    --name face-attribute-api \
    -p 8000:8000 \
    -v "$(pwd)/deployment/images:/app/data" \
    face-attribute-api

echo ""
echo "Step 5: Waiting for API to be ready..."
sleep 10

echo ""
echo "Step 6: Testing health endpoint..."
curl -s http://localhost:8000/health | python3 -m json.tool || echo "Health check failed"

echo ""
echo "======================================"
echo "✓ Deployment complete!"
echo "======================================"
echo ""
echo "Access the application at:"
echo "  - Web Interface: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/health"
echo ""
echo "To view logs:"
echo "  docker logs -f face-attribute-api"
echo ""
echo "To stop:"
echo "  docker stop face-attribute-api"
echo ""
echo "To export image:"
echo "  docker save -o api.tar face-attribute-api"
echo ""
