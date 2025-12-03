#!/bin/bash
# Script d'initialisation d'Apache Airflow pour le projet MLOps

set -e

echo "======================================"
echo "  Initialisation Apache Airflow"
echo "======================================"

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker n'est pas installé"
    echo "   Veuillez installer Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose n'est pas installé"
    echo "   Veuillez installer Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker et Docker Compose sont installés"

# Créer le fichier .env si nécessaire
if [ ! -f .env ]; then
    echo ""
    echo "Création du fichier .env..."
    cp .env.example .env
    
    # Détecter l'UID de l'utilisateur sur Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        USER_ID=$(id -u)
        sed -i "s/AIRFLOW_UID=50000/AIRFLOW_UID=$USER_ID/" .env
        echo "✓ AIRFLOW_UID configuré à $USER_ID"
    fi
fi

# Créer les dossiers nécessaires
echo ""
echo "Création des dossiers Airflow..."
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config
mkdir -p metrics plots models logs artifacts

echo "✓ Dossiers créés"

# Initialiser la base de données Airflow
echo ""
echo "Initialisation de la base de données Airflow..."
echo "  (Cela peut prendre quelques minutes...)"
docker-compose up airflow-init

echo ""
echo "======================================"
echo "  Airflow initialisé avec succès!"
echo "======================================"
echo ""
echo "Pour démarrer Airflow:"
echo "  docker-compose up -d"
echo ""
echo "Pour accéder à l'interface web:"
echo "  http://localhost:8080"
echo "  Username: airflow"
echo "  Password: airflow"
echo ""
echo "Pour arrêter Airflow:"
echo "  docker-compose down"
echo ""
