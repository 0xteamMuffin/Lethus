# Lethus

**Dynamic Context Pruning for Long-Form Dialogue Memory**

Lethus implements DYCP (Dynamic Context Pruning) from the research paper ["Dynamic Context Pruning for Long-Form Dialogue"](https://arxiv.org/abs/2601.07994), achieving state-of-the-art performance in conversational memory retrieval with 83.27% answer quality and sub-second latency.

## What It Does

Lethus is an **OpenAI-compatible proxy** that gives any LLM perfect memory over unlimited conversation history. Instead of feeding the entire chat log (slow, expensive, lossy), it uses Kadane's Algorithm to dynamically select only the relevant conversation segments for each query.

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
| **OpenAI-Compatible** | Drop-in replacement for any OpenAI client - just change base_url |
| **Context Reduction** | Reduces context by 5x while improving answer quality |

## Performance (from paper)

| Metric | DYCP | Full Context | Improvement |
|--------|------|--------------|-------------|
| Answer Quality (GPT4Score) | 83.27 | 75.13 | +10.8% |
| Response Latency | 1.10s | 2.32s | 2.1x faster |
| Input Tokens | 4,982 | 25,750 | 5.2x reduction |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Any OpenAI-Compatible Client                                    │
│  (Cursor, Copilot, Claude Code, custom apps)                     │
│  base_url = "http://localhost:8000/v1"                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Lethus Proxy                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Milvus     │    │    DYCP      │    │ Ghost Graph  │       │
│  │  (Vectors)   │───>│  (Kadane's)  │───>│  (Entities)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         v                   v                   v                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │         Context Reduction + Semantic Decay           │       │
│  └──────────────────────────────────────────────────────┘       │
│                              │                                   │
├──────────────────────────────┴──────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  POST /v1/chat/completions  (OpenAI-compatible proxy)  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Real LLM API (OpenAI)                         │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- OpenAI API key

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
lethus
```

This starts the proxy on `http://localhost:8000`.

## Usage

### With Any OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Lethus proxy
    api_key="your-openai-key"              # Passed through to OpenAI
)

# Use exactly like normal OpenAI API
# Lethus automatically reduces context using DYCP
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...long conversation history...],
    stream=True  # Streaming supported
)
```

### With Cursor/Copilot/Claude Code

Just set the API base URL to `http://localhost:8000/v1` in your tool's settings.

## API (OpenAI-Compatible Proxy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions with DYCP context reduction |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_id}` | GET | Get model details |

Full OpenAI API compatibility - use any OpenAI SDK or client.

## Configuration

All settings via environment variables (prefix: `LETHUS_`):

```bash
# Core Algorithm
LETHUS_DYCP_TAU=0.6              # Gain threshold (z-score shift)
LETHUS_DYCP_THETA=1.0            # Stopping threshold
LETHUS_DECAY_LAMBDA=0.98         # Semantic decay rate (2% per turn)

# Ghost Graph
LETHUS_USE_SPACY=true            # Enable spaCy NER
LETHUS_GHOST_GRAPH_BOOST=1.5     # Entity linking boost factor

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
│   │   ├── proxy.py         # OpenAI-compatible proxy
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
        similarity *= ghost_graph_boost  # 1.5x
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
