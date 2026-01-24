# Lethus AI - DYCP Memory System

A sophisticated conversational AI system with DYCP (Do You Copy?) memory retrieval, enabling long-term context awareness and intelligent conversation management.

## 🎯 Features

### Memory Flow Pipeline

```
New turn
   ↓
Importance detection → pinned memory
   ↓
Turn pairing
   ↓
Query embedding (+ variants)
   ↓
Weighted relevance scoring
   ↓
DYCP span selection
   ↓
Span merging & gap filling
   ↓
Token-aware trimming
   ↓
Confidence check & fallback
   ↓
Prompt assembly
   ↓
LLM
```

### Key Capabilities

- 🧠 **Intelligent Memory Retrieval**: DYCP algorithm for contextual conversation spans
- 📌 **Automatic Importance Detection**: Identifies and pins critical information
- 🔍 **Multi-Query Enhancement**: Query variants for better retrieval accuracy
- ⚖️ **Token Budget Management**: Smart context trimming to fit model limits
- 🎯 **Confidence-Based Fallback**: Falls back to recent context when needed
- 🔑 **User API Keys**: Bring your own OpenAI key for full control

## 📁 Project Structure

```
lethus-ai/
├── memory-backend/          # FastAPI backend with DYCP memory
│   ├── src/
│   │   ├── main.py         # FastAPI application
│   │   ├── memory_orchestrator.py  # Main memory pipeline
│   │   ├── importance.py   # Importance detection & embeddings
│   │   ├── retrieval.py    # DYCP span selection
│   │   ├── context_processing.py  # Token trimming & confidence
│   │   ├── database.py     # PostgreSQL models
│   │   ├── milvus_client.py # Vector database client
│   │   └── config.py       # Configuration
│   ├── docker-compose.yml  # Postgres + Milvus
│   ├── setup.sh           # Automated setup script
│   └── README.md
├── fe/                     # Next.js frontend
│   ├── app/
│   ├── components/
│   │   └── mainpage.tsx   # Main chat interface
│   ├── api/
│   │   ├── http.ts
│   │   └── message.ts     # Backend API integration
│   └── package.json
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.10+ (for backend)
- **Docker & Docker Compose** (for databases)
- **OpenAI API Key** (for LLM and embeddings)

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd memory-backend
   ```

2. **Run the automated setup**:
   ```bash
   ./setup.sh
   ```
   
   This will:
   - Start PostgreSQL and Milvus in Docker
   - Install Python dependencies
   - Initialize the database
   - Create `.env` file

3. **Start the backend server**:
   ```bash
   # With Poetry
   poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   
   # Or with Python directly
   python -m src.main
   ```

4. **Verify backend is running**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd fe
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   ```
   http://localhost:3000
   ```

### First Run

1. When you first open the frontend, you'll be prompted to enter your OpenAI API key
2. Get your API key from: https://platform.openai.com/api-keys
3. The key is stored locally in your browser (never sent to our servers)
4. Start chatting! The system will automatically:
   - Create embeddings for your conversations
   - Detect important information
   - Retrieve relevant context for each message
   - Provide memory-enhanced responses

## 🔧 Configuration

### Backend Configuration

Edit `memory-backend/.env`:

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=memory_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Vector DB
MILVUS_HOST=localhost
MILVUS_PORT=19530

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# Memory Settings (in src/config.py)
embedding_dim=1536           # OpenAI ada-002
dycp_window_size=10          # Conversation span window
max_context_tokens=8000      # Max tokens for context
importance_threshold=0.7     # Pinning threshold
```

### Frontend Configuration

Create `fe/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📖 API Documentation

### Chat Endpoint

**POST** `/api/chat`

Send a message and get a memory-enhanced response.

```json
{
  "conversation_id": 1,      // optional, creates new if not provided
  "user_id": "user123",
  "message": "What did we discuss about Python?",
  "openai_api_key": "sk-..."
}
```

**Response**:
```json
{
  "conversation_id": 1,
  "turn_id": 42,
  "message": "Based on our previous discussions...",
  "retrieved_context": {
    "spans": [...],          // Retrieved conversation spans
    "pinned_memories": [...], // Important pinned information
    "confidence": {
      "confident": true,
      "score": 0.85
    }
  },
  "metadata": {
    "importance_score": 0.75,
    "is_pinned": true,
    "turn_number": 42
  }
}
```

### Other Endpoints

- `POST /api/conversations` - Create conversation
- `GET /api/conversations/{id}` - Get conversation details
- `GET /api/conversations/user/{user_id}` - List user conversations
- `GET /api/conversations/{id}/turns` - Get conversation history
- `DELETE /api/conversations/{id}` - Delete conversation

Full API docs: http://localhost:8000/docs

## 🏗️ Architecture Details

### Memory Components

1. **Importance Detector**: Analyzes conversations for important information using keyword matching and semantic analysis

2. **Embedding Generator**: Creates vector embeddings using OpenAI's text-embedding-ada-002 model

3. **Relevance Scorer**: Multi-query retrieval with query variants for better accuracy

4. **DYCP Span Selector**: Selects conversation windows around relevant turns, merges overlapping spans

5. **Context Trimmer**: Token-aware trimming to fit within model context limits

6. **Confidence Checker**: Validates retrieval quality, falls back to recent context if needed

### Data Flow

1. User sends message → Backend receives with API key
2. System retrieves pinned memories from Postgres
3. Generates embeddings for query + variants
4. Searches Milvus for similar conversation turns
5. Selects and merges conversation spans (DYCP)
6. Trims context to fit token budget
7. Checks confidence, applies fallback if needed
8. Assembles prompt with context
9. Calls OpenAI for response
10. Stores turn, calculates importance, updates embeddings
11. Returns response with metadata

## 🐳 Docker Services

The `docker-compose.yml` provides:

- **PostgreSQL**: Conversation storage, turn history, pinned memories
- **Milvus**: Vector embeddings for semantic search

To manage services:

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Reset data
docker-compose down -v
```

## 🧪 Testing

```bash
cd memory-backend
poetry run pytest
```

## 📝 Development

### Adding New Features

1. Backend changes: Update relevant modules in `src/`
2. Database changes: Modify models in `database.py`
3. Frontend changes: Update components in `fe/components/`
4. API changes: Update `main.py` and `api/message.ts`

### Code Style

Backend:
```bash
poetry run black src/
poetry run ruff check src/
```

Frontend:
```bash
npm run lint
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- DYCP (Do You Copy?) memory retrieval algorithm
- OpenAI for embeddings and LLM
- Milvus for vector database
- FastAPI for backend framework
- Next.js for frontend framework

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Check the API docs at `/docs`
- Review the README files in each directory

---

Built with ❤️ using FastAPI, Milvus, PostgreSQL, and Next.js
