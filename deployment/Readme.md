Face Attribute Classification - Système de Déploiement
📋 Description
Système complet de déploiement (API + Docker + Frontend Web) pour la recherche d'images par attributs faciaux. L'application utilise un modèle de deep learning (best_model.pth) pour prédire les attributs faciaux et permet de rechercher des images selon ces attributs.

🎯 Attributs Détectés
Le modèle best_model.pth prédit les attributs suivants :

Barbe → (0: non, 1: oui) - binaire
Moustache → (0: non, 1: oui) - binaire
Lunettes → (0: non, 1: oui) - binaire
Taille des cheveux → (0: chauve, 1: court, 2: long) - 3 classes
Couleur des cheveux → (0: blond, 1: châtain, 2: roux, 3: brun, 4: gris/blanc) - 5 classes
🏗️ Architecture
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
🚀 API Endpoints
GET /
Page d'accueil avec l'interface web de recherche

GET /health
Health check de l'API

{
 "status": "healthy",
 "model_loaded": true,
 "images_scanned": 1500
}
GET /api/attributes
Liste des attributs disponibles et leurs valeurs possibles

{
 "barbe": [{"value": 0, "label": "Non"}, {"value": 1, "label": "Oui"}],
 "moustache": [...],
 ...
}
POST /api/search
Recherche d'images par attributs sélectionnés

Request:

{
 "barbe": [1],
 "moustache": [0, 1],
 "lunettes": null,
 "taille_cheveux": [1, 2],
 "couleur_cheveux": [3]
}
Response:

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
GET /api/images/{filename}
Récupérer une image depuis le répertoire de données

POST /api/predict
Prédire les attributs d'une image uploadée

Request: multipart/form-data avec un fichier image

Response:

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
