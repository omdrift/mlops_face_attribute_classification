FROM python:3.10-slim

WORKDIR /app

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best_model.pth \
    DATA_DIR=/app/data

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copier l'API et le frontend
COPY api/ ./api/
COPY frontend/ ./frontend/

# Créer le dossier src/models
RUN mkdir -p ./src/models

# Copier l'architecture du modèle
COPY models/architecture.py ./src/models/architecture.py

# Créer les fichiers __init__.py
RUN echo '"""Models package"""' > ./src/__init__.py
RUN echo '"""Models package"""' > ./src/models/__init__.py

# Copier le modèle entraîné
COPY models/best_model.pth ./models/best_model.pth

# Créer le dossier data (sera monté en volume)
RUN mkdir -p /app/data

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]