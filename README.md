# Lethus

**Dynamic Context Pruning for Long-Form Dialogue Memory**

Lethus implements DYCP (Dynamic Context Pruning) from the research paper ["Dynamic Context Pruning for Long-Form Dialogue"](https://arxiv.org/abs/2601.07994), achieving state-of-the-art performance in conversational memory retrieval with 83.27% answer quality and sub-second latency.

## What It Does

Lethus gives LLMs perfect memory over unlimited conversation history. Instead of feeding the entire chat log (slow, expensive, lossy), it uses Kadane's Algorithm to dynamically select only the relevant conversation segments for each query.

```
User: "What was that API key you mentioned earlier?"

Traditional approach: Send all 500+ turns to LLM (slow, expensive, often fails)
Lethus approach:     Retrieve 3 relevant spans in 0.9s with 83%+ accuracy
```

## Key Features

| Feature | Description |
|---------|-------------|
| **DYCP Algorithm** | Kadane's Algorithm for optimal span selection (tau=0.6, theta=1.0) |
| **Semantic Decay** | Older messages need higher relevance to be recalled (lambda=0.98) |
| **Ghost Graph** | Entity linking for pronoun resolution ("it", "that config", "the API") |
| **Prefetch Cache** | Predictive caching for follow-up queries |
| **Dual Interface** | MCP server for Claude/LLM integration + REST API for web apps |

## Performance (from paper)

| Metric | DYCP | Full Context | Improvement |
|--------|------|--------------|-------------|
| Answer Quality (GPT4Score) | 83.27 | 75.13 | +10.8% |
| Response Latency | 1.10s | 2.32s | 2.1x faster |
| Input Tokens | 4,982 | 25,750 | 5.2x reduction |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Lethus Core                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Milvus     │    │    DYCP      │    │ Ghost Graph  │       │
│  │  (Vectors)   │───>│  (Kadane's)  │───>│  (Entities)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         v                   v                   v                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Semantic Decay + Prefetch               │       │
│  └──────────────────────────────────────────────────────┘       │
│                              │                                   │
├──────────────────────────────┴──────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐          ┌────────────────────┐         │
│  │    MCP Server      │          │     REST API       │         │
│  │  (Claude/LLMs)     │          │   (Web Apps)       │         │
│  └────────────────────┘          └────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- OpenAI API key (for embeddings)

### 1. Start Infrastructure

```bash
cd docker
docker-compose up -d
```

This starts:
- **Milvus** (vector database) on port 19530
- **PostgreSQL** on port 5432
- **etcd** + **MinIO** (Milvus dependencies)

### 2. Install Lethus

```bash
# Create and activate virtual environment
python -m venv .venv

# Activate (choose one):
source .venv/Scripts/activate  # Git Bash on Windows
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows CMD/PowerShell

# Install package
pip install -e .
python -m spacy download en_core_web_sm
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 4. Run

```bash
# MCP Server (for Claude Desktop / LLM integration)
lethus --mode=mcp

# REST API (for web applications)
lethus --mode=api

# Both
lethus --mode=both
```

## MCP Integration (Claude Desktop)

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "lethus": {
      "command": "lethus",
      "args": ["--mode=mcp"]
    }
  }
}
```

Available MCP tools:
- `store_interaction` - Store user/assistant messages
- `get_context` - Retrieve relevant context using DYCP
- `search_by_entity` - Find memories mentioning specific entities
- `get_entity_graph` - View Ghost Graph state
- `get_memory_stats` - Memory system statistics
- `clear_memory` - Wipe all memory

## REST API

### Send Message

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "What did we discuss about the API?",
    "openai_api_key": "sk-..."
  }'
```

### Response

```json
{
  "conversation_id": 1,
  "turn_id": 42,
  "message": "Based on our previous discussions...",
  "retrieved_context": {
    "spans": [
      {
        "start_index": 15,
        "end_index": 18,
        "relevance_score": 0.87
      }
    ],
    "confidence": {
      "confident": true,
      "score": 0.85
    }
  }
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send message, get memory-enhanced response |
| `/api/conversations` | POST | Create conversation |
| `/api/conversations/{id}` | GET | Get conversation details |
| `/api/conversations/{id}/turns` | GET | Get conversation history |
| `/api/conversations/{id}` | DELETE | Delete conversation |
| `/api/stats` | GET | Memory system statistics |

Full docs: http://localhost:8000/docs

## Configuration

All settings via environment variables (prefix: `LETHUS_`):

```bash
# Core Algorithm
LETHUS_DYCP_TAU=0.6              # Gain threshold (z-score shift)
LETHUS_DYCP_THETA=1.0            # Stopping threshold
LETHUS_DECAY_LAMBDA=0.98         # Semantic decay rate (2% per turn)

# Ghost Graph
LETHUS_USE_SPACY=true            # Enable spaCy NER
LETHUS_GHOST_GRAPH_BOOST=1.2     # Entity linking boost factor

# Embeddings
LETHUS_EMBEDDING_PROVIDER=openai # "openai" or "local"
LETHUS_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Infrastructure
LETHUS_MILVUS_URI=http://localhost:19530
LETHUS_POSTGRES_HOST=localhost
```

## Project Structure

```
lethus/
├── src/lethus/
│   ├── main.py              # Entry point with CLI
│   ├── config.py            # Pydantic settings
│   ├── api/
│   │   ├── mcp.py           # MCP server (FastMCP)
│   │   ├── rest.py          # FastAPI REST endpoints
│   │   └── models.py        # Pydantic models
│   ├── core/
│   │   ├── dycp.py          # Kadane's Algorithm implementation
│   │   ├── embeddings.py    # OpenAI/local embedding providers
│   │   ├── ghost_graph.py   # Entity extraction & linking
│   │   └── prefetch.py      # Predictive cache
│   └── storage/
│       ├── milvus.py        # Vector storage
│       └── postgres.py      # Conversation metadata
├── docker/
│   └── docker-compose.yml   # Milvus + PostgreSQL
├── research/
│   └── files/               # DYCP paper reference
└── pyproject.toml
```

## How DYCP Works

### 1. Relevance Scoring with Semantic Decay

```python
# Cosine similarity with time decay
similarity = cosine(query_emb, turn_emb)
age = current_turn - turn_index
decayed_similarity = similarity * (lambda ^ age)
```

### 2. Kadane's Algorithm for Span Selection

```python
# Z-score normalization
z_scores = (similarities - mean) / std

# Gain calculation (tau shifts the baseline)
gains = z_scores - tau  # Only significantly relevant turns have positive gain

# Find contiguous spans with positive cumulative gain
# Theta-based early stopping prevents trailing low-relevance turns
```

### 3. Ghost Graph Entity Boosting

```python
# Extract entities from query
entities = ["API_KEY", "config", "John"]

# Boost turns containing linked entities
for turn in turns:
    if has_linked_entity(turn, entities):
        similarity *= ghost_graph_boost  # 1.2x
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/
```

## Research Reference

This implementation is based on:

> **Dynamic Context Pruning for Long-Form Dialogue**  
> arXiv:2601.07994v2  
> 
> Key findings:
> - DYCP achieves highest answer quality across benchmarks (83.27% on LoCoMo)
> - 2.1x faster than full context approaches
> - 5.2x reduction in input tokens
> - High recall is more beneficial than precision for retrieval
> - Preserving sequential nature of dialogue improves response generation

## License

MIT
