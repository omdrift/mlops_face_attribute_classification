#!/bin/bash
# Script to clean up Airflow completely
# Use this if you encounter persistent issues

set -e

echo "🧹 Cleaning up Airflow..."
echo ""

cd "$(dirname "$0")"

# Step 1: Stop all containers
echo "🛑 Step 1: Stopping containers..."
docker-compose -f docker-compose.airflow.yml down -v 2>/dev/null || true
echo "   Containers stopped"
echo ""

# Step 2: Remove local directories (optional - user confirmation)
read -p "🗑️  Remove local logs and plugins? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Removing logs, plugins, config..."
    rm -rf ./logs/* ./plugins/* ./config/*
    echo "   ✅ Cleaned"
else
    echo "   ⏭️  Skipped"
fi
echo ""

# Step 3: Remove Docker volumes
read -p "🗑️  Remove PostgreSQL volume (will lose all data)? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Removing volumes..."
    docker volume rm airflow_postgres-db-volume 2>/dev/null || true
    echo "   ✅ Removed"
else
    echo "   ⏭️  Skipped"
fi
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ Cleanup complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "💡 Next steps:"
echo "   Run: ./start_airflow.sh"
echo ""
