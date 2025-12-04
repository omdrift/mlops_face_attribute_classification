#!/bin/bash
# Script to properly initialize and start Airflow
# This handles all the setup steps to avoid init errors

set -e

echo "🚀 Starting Airflow Setup..."
echo ""

# Get the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Set AIRFLOW_UID
echo "📝 Step 1: Setting AIRFLOW_UID..."
export AIRFLOW_UID=$(id -u)
echo "   AIRFLOW_UID set to: $AIRFLOW_UID"
echo ""

# Step 2: Create required directories
echo "📁 Step 2: Creating required directories..."
mkdir -p ./dags ./logs ./plugins ./config
echo "   Directories created"
echo ""

# Step 3: Set permissions
echo "🔐 Step 3: Setting permissions..."
chmod -R 755 ./dags ./plugins ./config
chmod -R 777 ./logs  # Logs need write access from containers
echo "   Permissions set"
echo ""

# Step 4: Check if containers are already running
echo "🔍 Step 4: Checking for existing containers..."
if docker-compose -f docker-compose.airflow.yml ps | grep -q "Up"; then
    echo "   ⚠️  Containers already running. Stopping them first..."
    docker-compose -f docker-compose.airflow.yml down
    echo "   Containers stopped"
fi
echo ""

# Step 5: Start Airflow
echo "🚀 Step 5: Starting Airflow containers..."
echo "   This may take a few minutes on first run..."
echo ""

AIRFLOW_UID=$AIRFLOW_UID docker-compose -f docker-compose.airflow.yml up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Step 6: Check status
echo ""
echo "📊 Step 6: Checking container status..."
docker-compose -f docker-compose.airflow.yml ps
echo ""

# Step 7: Check if init was successful
if docker-compose -f docker-compose.airflow.yml ps | grep -q "airflow-init.*Exited (0)"; then
    echo "✅ Airflow initialization successful!"
elif docker-compose -f docker-compose.airflow.yml ps | grep -q "airflow-init.*Exited (1)"; then
    echo "❌ Airflow initialization failed!"
    echo ""
    echo "📋 Init container logs:"
    docker-compose -f docker-compose.airflow.yml logs airflow-init
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   1. Make sure PostgreSQL is healthy: docker-compose -f docker-compose.airflow.yml logs postgres"
    echo "   2. Check if port 5432 is available: lsof -i :5432"
    echo "   3. Try a clean start: ./cleanup_airflow.sh && ./start_airflow.sh"
    exit 1
else
    echo "⏳ Init container still running or not started yet..."
    echo "   Wait a moment and check: docker-compose -f docker-compose.airflow.yml logs airflow-init"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Airflow is starting up!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🌐 Access Airflow UI:"
echo "   URL: http://localhost:8080"
echo "   Username: airflow"
echo "   Password: airflow"
echo ""
echo "📝 Useful commands:"
echo "   View logs: docker-compose -f docker-compose.airflow.yml logs -f"
echo "   Stop: docker-compose -f docker-compose.airflow.yml down"
echo "   Restart: docker-compose -f docker-compose.airflow.yml restart"
echo ""
echo "⏳ Note: It may take 1-2 minutes for the webserver to be fully ready"
echo "════════════════════════════════════════════════════════════════"
