#!/bin/bash

echo "🛑 Stopping RecommendIt System..."

# Stop all services
docker-compose down

echo "✅ All services stopped"
echo "💡 To start again, run: ./start.sh"



