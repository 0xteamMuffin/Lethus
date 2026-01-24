# Quick Start Guide - In-Memory Mode

This is the fastest way to test the DYCP Memory System without Docker downloads!

## Setup (< 2 minutes)

### 1. Create Virtual Environment
```bash
cd memory-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **Milvus Lite** (in-memory, no Docker needed!)
- FastAPI, SQLAlchemy, OpenAI, etc.

### 3. Start Just PostgreSQL (optional Docker)
```bash
# Option A: With Docker (recommended)
docker run -d --name memory-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=memory_db \
  -p 5432:5432 \
  postgres:15-alpine

# Option B: Use existing PostgreSQL
# Just make sure it's running on localhost:5432
```

### 4. Initialize Database
```bash
python -c "from src.database import init_db; init_db()"
```

### 5. Start Backend
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start Frontend (in another terminal)
```bash
cd ../fe
npm run dev
```

## Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## How It Works

**In-Memory Mode:**
- ✅ Milvus Lite runs in-memory (no Docker)
- ✅ Starts instantly
- ✅ Perfect for testing
- ⚠️ Data cleared on restart (development only)

**Full Docker Mode:**
- Uses docker-compose for both Postgres + Milvus
- Persistent data
- Production-ready

## Configuration

`.env` file controls the mode:

```env
# Use in-memory Milvus (fast, no Docker)
MILVUS_MODE=local

# Use Docker Milvus (persistent)
# MILVUS_MODE=remote
# MILVUS_HOST=localhost
# MILVUS_PORT=19530
```

## Troubleshooting

### Python 3.13 on Arch Linux?
The virtual environment handles the "externally managed" error automatically.

### Can't install psycopg2-binary?
Install PostgreSQL dev headers:
```bash
# Arch
sudo pacman -S postgresql-libs

# Ubuntu/Debian
sudo apt-get install libpq-dev

# macOS
brew install postgresql
```

### Need to switch to full Docker mode?
1. Edit `.env`: set `MILVUS_MODE=remote`
2. Run `docker-compose up -d`
3. Restart backend

## Next Steps

1. Open http://localhost:3000
2. Enter your OpenAI API key
3. Start chatting!
4. Watch the DYCP memory flow in action

The system will automatically:
- Detect important information
- Create embeddings in Milvus Lite
- Retrieve relevant context
- Provide memory-enhanced responses
