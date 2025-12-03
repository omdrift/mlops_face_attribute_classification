# Face Attribute Classification - Système de Déploiement

## 📋 Description

Système complet de déploiement (API + Docker + Frontend Web) pour la recherche d'images par attributs faciaux. L'application utilise un modèle de deep learning (`best_model.pth`) pour prédire les attributs faciaux et permet de rechercher des images selon ces attributs.

## 🎯 Attributs Détectés

Le modèle `best_model.pth` prédit les attributs suivants :

- **Barbe** → (0: non, 1: oui) - binaire
- **Moustache** → (0: non, 1: oui) - binaire
- **Lunettes** → (0: non, 1: oui) - binaire
- **Taille des cheveux** → (0: chauve, 1: court, 2: long) - 3 classes
- **Couleur des cheveux** → (0: blond, 1: châtain, 2: roux, 3: brun, 4: gris/blanc) - 5 classes

## 🏗️ Architecture

```
deployment/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── inference.py         # Model inference logic
│   └── utils.py             # Utility functions
├── frontend/
│   ├── templates/
│   │   └── index.html       # Interface web avec filtres d'attributs
│   └── static/
│       ├── css/
│       │   └── style.css    # Styles CSS modernes
│       └── js/
│           └── app.js       # JavaScript pour interactions
├── Dockerfile
├── docker-compose.yml
├── requirements-api.txt
├── .dockerignore
└── README.md
```

## 🚀 API Endpoints

### GET /
Page d'accueil avec l'interface web de recherche

### GET /health
Health check de l'API
```json
{
  "status": "healthy",
  "model_loaded": true,
  "images_scanned": 1500
}
```

### GET /api/attributes
Liste des attributs disponibles et leurs valeurs possibles
```json
{
  "barbe": [{"value": 0, "label": "Non"}, {"value": 1, "label": "Oui"}],
  "moustache": [...],
  ...
}
```

### POST /api/search
Recherche d'images par attributs sélectionnés

**Request:**
```json
{
  "barbe": [1],
  "moustache": [0, 1],
  "lunettes": null,
  "taille_cheveux": [1, 2],
  "couleur_cheveux": [3]
}
```

**Response:**
```json
{
  "total": 42,
  "images": [
    {
      "filename": "image001.jpg",
      "path": "/api/images/image001.jpg",
      "attributes": {
        "barbe": 1,
        "moustache": 0,
        "lunettes": 1,
        "taille_cheveux": 2,
        "couleur_cheveux": 3
      }
    },
    ...
  ]
}
```

### GET /api/images/{filename}
Récupérer une image depuis le répertoire de données

### POST /api/predict
Prédire les attributs d'une image uploadée

**Request:** multipart/form-data avec un fichier image

**Response:**
```json
{
  "barbe": 1,
  "moustache": 0,
  "lunettes": 1,
  "taille_cheveux": 2,
  "couleur_cheveux": 3,
  "labels": {
    "barbe": "Oui",
    "moustache": "Non",
    "lunettes": "Oui",
    "taille_cheveux": "Long",
    "couleur_cheveux": "Brun"
  }
}
```

## 🖥️ Interface Web

L'interface web offre :
- **Filtres interactifs** pour chaque attribut facial
- **Sélection multiple** par attribut (logique OR au sein d'un attribut)
- **Grille d'images responsive** avec les résultats
- **Pagination** automatique (20 images par page)
- **Upload d'image** pour prédiction instantanée
- **Design moderne** et responsive (mobile-friendly)

## 📦 Build de l'Image Docker

### Prérequis

Assurez-vous que le modèle `models/best_model.pth` existe à la racine du projet.

### Build

```bash
cd /chemin/vers/mlops_face_attribute_classification

# Build l'image Docker
docker build -t face-attribute-api -f deployment/Dockerfile .
```

## 💾 Export de l'Image (api.tar)

```bash
# Exporter l'image en fichier .tar (sans les données)
docker save -o api.tar face-attribute-api

# Vérifier la taille du fichier
ls -lh api.tar
```

Le fichier `api.tar` contient l'image Docker complète avec :
- ✅ Le modèle `best_model.pth`
- ✅ Tout le code de l'API
- ✅ Les dépendances Python
- ❌ Pas les images (montées via volume)

## 📥 Import et Déploiement

### 1. Charger l'image Docker

```bash
# Sur la machine de destination
docker load -i api.tar
```

### 2. Préparer les données

Créez un répertoire local contenant vos images :

```bash
mkdir -p /chemin/local/images
# Copiez vos images dans ce répertoire
```

### 3. Lancer le conteneur

#### Méthode 1 : Docker Run

```bash
docker run -d \
  --name face-attribute-api \
  -p 8000:8000 \
  -v /chemin/local/images:/app/data \
  face-attribute-api
```

#### Méthode 2 : Docker Compose

Modifiez le chemin du volume dans `docker-compose.yml` :

```yaml
volumes:
  - /chemin/local/images:/app/data
```

Puis lancez :

```bash
cd deployment
docker-compose up -d
```

### 4. Accéder à l'application

Ouvrez votre navigateur et accédez à :
- **Interface Web** : http://localhost:8000
- **Health Check** : http://localhost:8000/health
- **API Docs** : http://localhost:8000/docs (Swagger UI)

## 🔧 Configuration

### Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `PORT` | Port d'écoute de l'API | 8000 |
| `DATA_DIR` | Répertoire des images | /app/data |
| `MODEL_PATH` | Chemin du modèle | /app/models/best_model.pth |

### Volume de données

Le répertoire `/app/data` dans le conteneur doit être monté avec vos images locales :

```bash
-v /votre/chemin/local:/app/data
```

## 📊 Performances

- **Premier démarrage** : Le modèle scanne toutes les images et met en cache les prédictions (~1-2 secondes par 100 images)
- **Recherches suivantes** : Instantanées (utilise le cache)
- **Prédiction d'une nouvelle image** : ~100ms

## 🛠️ Développement Local

### Sans Docker

```bash
# Installer les dépendances
pip install -r deployment/requirements-api.txt

# Lancer l'API
cd deployment
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Avec Docker

```bash
# Build et run en mode dev
docker-compose up --build
```

## 🧪 Tests de l'API

### Avec cURL

```bash
# Health check
curl http://localhost:8000/health

# Liste des attributs
curl http://localhost:8000/api/attributes

# Recherche d'images
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"barbe": [1], "lunettes": [1]}'

# Prédiction d'une image
curl -X POST http://localhost:8000/api/predict \
  -F "file=@/chemin/vers/image.jpg"
```

### Avec Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Recherche
search_params = {
    "barbe": [1],
    "moustache": [0],
    "lunettes": [1]
}
response = requests.post("http://localhost:8000/api/search", json=search_params)
results = response.json()
print(f"Trouvé {results['total']} images")

# Prédiction
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"file": f}
    )
print(response.json())
```

## 🐛 Dépannage

### Le conteneur ne démarre pas

```bash
# Vérifier les logs
docker logs face-attribute-api

# Vérifier que le modèle existe
docker exec face-attribute-api ls -l /app/models/best_model.pth
```

### Les images ne s'affichent pas

```bash
# Vérifier le montage du volume
docker exec face-attribute-api ls /app/data

# Vérifier les permissions
docker exec face-attribute-api ls -la /app/data
```

### Le modèle ne charge pas

- Vérifiez que `best_model.pth` existe dans le répertoire `models/`
- Vérifiez la compatibilité de la version de PyTorch
- Consultez les logs : `docker logs face-attribute-api`

### Erreur de mémoire

Si vous avez beaucoup d'images, augmentez la mémoire allouée au conteneur :

```bash
docker run -d \
  --name face-attribute-api \
  --memory=4g \
  -p 8000:8000 \
  -v /chemin/local/images:/app/data \
  face-attribute-api
```

## 📝 Notes Importantes

- ⚠️ L'image Docker **ne contient pas** les images de données (montées via volume)
- ✅ Le modèle `best_model.pth` **est inclus** dans l'image Docker
- 🔄 Les prédictions sont mises en cache pour améliorer les performances
- 🎨 L'interface web permet la sélection **multiple** de valeurs par attribut (logique OR)
- 📱 L'interface est **responsive** et fonctionne sur mobile

## 📜 Licence

Ce projet est fourni à des fins éducatives.

## 👥 Auteurs

Équipe MLOps Face Attribute Classification

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt GitHub.
