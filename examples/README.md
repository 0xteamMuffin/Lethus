# Lethus Examples

Interactive examples to understand how the DYCP memory system works.

## Quick Start

```bash
cd examples
python 01_basic_dycp.py
```

## Examples

| File | Description | Requires Server? |
|------|-------------|------------------|
| `01_basic_dycp.py` | Core Kadane's Algorithm for span selection | No |
| `02_semantic_decay.py` | How older messages are deprioritized | No |
| `03_ghost_graph.py` | Entity extraction and pronoun resolution | No |
| `04_full_pipeline.py` | End-to-end DYCP processing | No |
| `05_api_usage.py` | REST API endpoint examples | Yes |
| `06_mcp_tools.py` | MCP tools reference for LLM integration | No |
| `07_comparison_demo.py` | DYCP vs Full Context comparison | No |

## Running Examples

### Standalone Examples (01-04, 06-07)

These run without any server:

```bash
python 01_basic_dycp.py
python 02_semantic_decay.py
python 03_ghost_graph.py
python 04_full_pipeline.py
python 06_mcp_tools.py
python 07_comparison_demo.py
```

### API Example (05)

Requires Lethus server running:

```bash
# Terminal 1: Start infrastructure
cd ../docker && docker-compose up -d

# Terminal 2: Start Lethus
lethus --mode=api

# Terminal 3: Run example
python 05_api_usage.py
```

## Example Output

### 01_basic_dycp.py

```
============================================================
DYCP EXAMPLE: Finding Relevant Conversation Spans
============================================================

Query: 'What database did we configure?'

Similarity scores per turn:
  Turn  0: 0.20 [LOW]
  Turn  1: 0.15 [LOW]
  Turn  2: 0.30 [LOW]
  Turn  3: 0.85 [HIGH]  ← PostgreSQL discussion
  Turn  4: 0.78 [HIGH]  ← Schema setup
  Turn  5: 0.72 [HIGH]  ← Connection string
  ...

SELECTED SPANS:
  Span 1: Turns 3 to 5 (the DB configuration block)
  Span 2: Turn 9 (later port mention)

EFFICIENCY:
  Total turns: 11
  Retrieved: 4
  Reduction: 64%
```

## Key Concepts Demonstrated

### 1. Kadane's Algorithm (01, 04)

The algorithm finds **contiguous spans** with positive cumulative gain:
- Z-score normalize relevance scores
- Subtract threshold (tau=0.6)
- Find spans where cumulative gain stays positive
- Stop when drop from peak exceeds theta (1.0)

### 2. Semantic Decay (02)

Formula: `score = similarity × (0.98 ^ age)`

| Age (turns) | Decay Factor | Effect |
|-------------|--------------|--------|
| 0 | 1.000 | Full weight |
| 50 | 0.364 | Moderate penalty |
| 200 | 0.018 | Heavy penalty |

### 3. Ghost Graph (03)

- Extracts entities: `AWS_ACCESS_KEY`, `DATABASE_URL`, URLs, IPs
- Links co-occurring entities
- Boosts similarity for turns with linked entities
- Helps resolve "that config" → specific entity

### 4. Full Pipeline (04)

1. Extract entities from all turns
2. Generate embeddings
3. Compute decayed similarities
4. Apply Ghost Graph boosting
5. Run Kadane's for span selection
6. Format context for LLM

## Why DYCP Works (07)

The comparison demo shows:
- Full context: 31 turns, ~400 tokens, 75% accuracy
- DYCP: 3 turns, ~40 tokens, 83% accuracy

**Less noise = Better answers**
