#!/bin/bash

echo "🚀 Starting DYCP Memory Backend Setup..."

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env file created (configured for in-memory Milvus)"
fi

# Ask user if they want to use Docker or in-memory mode
echo ""
echo "Choose setup mode:"
echo "1) Quick Start (PostgreSQL in Docker, Milvus in-memory) - Recommended"
echo "2) Full Docker (PostgreSQL + Milvus in Docker)"
echo ""
read -p "Enter choice (1 or 2) [default: 1]: " choice
choice=${choice:-1}

if [ "$choice" == "2" ]; then
    # Full Docker mode
    if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker/docker-compose not installed. Falling back to quick start mode."
        choice="1"
    else
        echo "🐳 Starting PostgreSQL and Milvus containers..."
        docker-compose up -d
        echo "⏳ Waiting for services to be ready..."
        sleep 10
        # Update .env for remote Milvus
        sed -i 's/MILVUS_MODE=local/MILVUS_MODE=remote/' .env
    fi
fi

if [ "$choice" == "1" ]; then
    # Quick start - only PostgreSQL in Docker
    echo "🚀 Quick Start Mode: Using in-memory Milvus..."
    
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        echo "🐳 Starting PostgreSQL container..."
        docker-compose up -d postgres
        echo "⏳ Waiting for PostgreSQL to be ready..."
        sleep 5
    else
        echo "⚠️  Docker not available. Make sure PostgreSQL is running on localhost:5432"
    fi
fi

# Check if poetry is installed
if command -v poetry &> /dev/null; then
    echo "📦 Installing Python dependencies with Poetry..."
    poetry install
    
    echo "🗄️  Initializing database..."
    poetry run python -c "from src.database import init_db; init_db()"
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "📊 Configuration:"
    echo "  Mode: $([ "$choice" == "1" ] && echo "Quick Start (In-memory Milvus)" || echo "Full Docker")"
    echo "  Backend API: http://localhost:8000"
    echo "  PostgreSQL: localhost:5432"
    if [ "$choice" == "2" ]; then
        echo "  Milvus: localhost:19530"
    else
        echo "  Milvus: In-memory (no Docker needed)"
    fi
    echo ""
    echo "To start the backend server, run:"
    echo "  poetry run python -m src.main"
    echo ""
    echo "Or:"
    echo "  poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
    
else
    echo "📦 Poetry not found. Installing dependencies with pip..."
    pip install -r requirements.txt
    
    echo "🗄️  Initializing database..."
    python -c "from src.database import init_db; init_db()"
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "📊 Configuration:"
    echo "  Mode: $([ "$choice" == "1" ] && echo "Quick Start (In-memory Milvus)" || echo "Full Docker")"
    echo "  Backend API: http://localhost:8000"
    echo "  PostgreSQL: localhost:5432"
    if [ "$choice" == "2" ]; then
        echo "  Milvus: localhost:19530"
    else
        echo "  Milvus: In-memory (no Docker needed)"
    fi
    echo ""
    echo "To start the backend server, run:"
    echo "  python -m src.main"
    echo ""
    echo "Or:"
    echo "  uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
fi

echo ""
echo "🌐 Once started, visit:"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
if [ "$choice" == "1" ]; then
    echo "To stop PostgreSQL: docker-compose stop postgres"
else
    echo "To stop all services: docker-compose down"
fi
