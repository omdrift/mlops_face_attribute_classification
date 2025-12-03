FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY deployment/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY deployment/api/ ./deployment/api/
COPY models/ ./models/

# Expose ports
EXPOSE 8000

# Run the API
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
