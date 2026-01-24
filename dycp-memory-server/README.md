# DYCP Memory Server

This Model Context Protocol (MCP) server implements the **Dynamic Context Pruning (DYCP)** algorithm described in the paper "Dynamic Context Pruning for Long-Form Dialogue".

## Core Logic

It uses **Kadane's Algorithm** on relevance scores to identify coherent spans of conversation to retain, rather than simple top-k retrieval or full context.

### Configuration (based on Section 5.4 of the paper)
- **Gain Threshold ($\tau$):** 0.6
- **Stopping Threshold ($\theta$):** 1.0
- **Normalization:** Z-score normalization of similarity scores.

## Architecture

1.  **Storage:** Local SQLite database to store conversation turns.
2.  **Embeddings:** Local `sentence-transformers` (default) or pluggable API.
3.  **Pruning:** On `get_context(query)`, it:
    - Embeds the query.
    - Calculates similarity with all history turns.
    - Applies DYCP to select spans.
    - Returns the formatted context.

## tools

- `store_turn(role, content)`: Saves a user/assistant message.
- `get_pruned_context(query)`: Retrieves the DYCP-optimized context.
