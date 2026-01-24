#!/bin/bash
# Fast install script for Python 3.13

cd /mnt/d-drive/Programming/work/lethus-ai/memory-backend
source venv/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing core packages..."
pip install fastapi uvicorn[standard] sqlalchemy python-dotenv

echo "📦 Installing pydantic (with Python 3.13 support)..."
pip install "pydantic>=2.10.0" "pydantic-settings>=2.6.0"

echo "📦 Installing database drivers..."
pip install "psycopg2-binary>=2.9.10"

echo "📦 Installing AI packages..."
pip install "openai>=1.54.0" "tiktoken>=0.8.0"

echo "📦 Installing data science packages..."
pip install "numpy>=2.0.0" "scikit-learn>=1.5.0"

echo "📦 Installing Milvus Lite (in-memory)..."
pip install "milvus>=2.4.0"

echo "📦 Installing remaining packages..."
pip install "python-multipart>=0.0.17"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Start PostgreSQL: docker run -d --name memory-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=memory_db -p 5432:5432 postgres:15-alpine"
echo "2. Initialize DB: python -c 'from src.database import init_db; init_db()'"
echo "3. Start server: uvicorn src.main:app --reload"
