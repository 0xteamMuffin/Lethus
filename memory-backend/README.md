# DYCP Memory Backend

A conversational memory system using DYCP (Do You Copy?) retrieval algorithm with FastAPI, Milvus vector database, and PostgreSQL.

## Features

- 🧠 **DYCP Memory Retrieval**: Intelligent context retrieval using conversation spans
- 📌 **Importance Detection**: Automatically identifies and pins important information
- 🔍 **Multi-Query Retrieval**: Enhanced relevance scoring with query variants
- ⚖️ **Token-Aware Context**: Smart trimming to fit within token budgets
- 🎯 **Confidence-Based Fallback**: Automatic fallback to recent context when needed
- 🔑 **User API Keys**: Bring your own OpenAI API key

## Architecture

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

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL
- Milvus (standalone or cluster)
- Poetry (recommended) or pip

### Installation

1. **Clone and navigate to the backend directory**:
   ```bash
   cd memory-backend
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   # or
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=memory_db
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   
   API_HOST=0.0.0.0
   API_PORT=8000
   CORS_ORIGINS=http://localhost:3000
   ```

4. **Start PostgreSQL**:
   ```bash
   # Using Docker
   docker run -d \
     --name postgres \
     -e POSTGRES_PASSWORD=postgres \
     -e POSTGRES_DB=memory_db \
     -p 5432:5432 \
     postgres:15
   ```

5. **Start Milvus**:
   ```bash
   # Using Docker
   docker run -d \
     --name milvus-standalone \
     -p 19530:19530 \
     -p 9091:9091 \
     -e ETCD_USE_EMBED=true \
     -e COMMON_STORAGETYPE=local \
     milvusdb/milvus:latest
   ```

6. **Run the application**:
   ```bash
   poetry run python -m src.main
   # or
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### Conversations

- `POST /api/conversations` - Create a new conversation
- `GET /api/conversations/{conversation_id}` - Get conversation details
- `GET /api/conversations/user/{user_id}` - Get all user conversations
- `GET /api/conversations/{conversation_id}/turns` - Get conversation history
- `DELETE /api/conversations/{conversation_id}` - Delete a conversation

### Chat

- `POST /api/chat` - Send a message with memory-enhanced response

**Request body**:
```json
{
  "conversation_id": 1,  // optional, creates new if not provided
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
  "message": "Based on our previous discussions about Python...",
  "retrieved_context": {
    "spans": [...],
    "pinned_memories": [...],
    "confidence": {...}
  },
  "metadata": {
    "importance_score": 0.75,
    "is_pinned": true,
    "turn_number": 42
  }
}
```

## Configuration

Key settings in `src/config.py`:

- `embedding_dim`: Embedding dimension (default: 1536 for OpenAI ada-002)
- `dycp_window_size`: Window size for span selection (default: 10)
- `max_context_tokens`: Maximum context tokens (default: 8000)
- `importance_threshold`: Threshold for pinning memories (default: 0.7)

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black src/
poetry run ruff check src/
```

## License

MIT
