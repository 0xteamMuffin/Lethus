"""
Lethus - DYCP Context Reduction Proxy

An OpenAI-compatible proxy that applies Dynamic Context Pruning
to reduce conversation history to only relevant spans.

Core components:
- DYCP: Kadane's Algorithm for optimal span selection
- Semantic Decay: Time-weighted relevance scoring
- Ghost Graph: Entity linking for pronoun resolution

Usage:
    Set base_url="http://localhost:8000/v1" in any OpenAI-compatible client.
"""
__version__ = "0.1.0"
