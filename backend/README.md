# Lethus Memory Backend

Conversational memory system with DYCP (Dynamic Context Pair) span selection.

## Features

- **Importance Detection**: Automatically identifies and pins important conversation turns
- **Turn Pairing**: Intelligently pairs related user-assistant exchanges
- **Query Embedding**: Generates embeddings with variants for better retrieval
- **Weighted Relevance Scoring**: Scores conversation spans by relevance
- **DYCP Span Selection**: Selects optimal conversation spans
- **Span Merging & Gap Filling**: Optimizes selected spans for coherence
- **Token-Aware Trimming**: Ensures context fits within token limits
- **Confidence Check & Fallback**: Validates context quality
- **LLM Integration**: Supports OpenAI models via user API keys

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Set up databases:
```bash
# Start Milvus (using docker)
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Start PostgreSQL
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the server:
```bash
poetry run uvicorn src.main:app --reload
```

## API Endpoints

- `POST /api/chat`: Send a message and get a response
- `POST /api/sessions`: Create a new conversation session
- `GET /api/sessions/{session_id}`: Get session history
- `POST /api/pin`: Pin an important turn
- `GET /api/health`: Health check

## Architecture

The system follows this flow:
1. New turn → Importance detection → Pinned memory
2. Turn pairing
3. Query embedding (+ variants)
4. Weighted relevance scoring
5. DYCP span selection
6. Span merging & gap filling
7. Token-aware trimming
8. Confidence check & fallback
9. Prompt assembly
10. LLM response
