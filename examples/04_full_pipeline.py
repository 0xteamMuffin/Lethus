"""
Example 4: Full DYCP Pipeline
End-to-end example showing how all components work together.
"""
import numpy as np
import sys
sys.path.insert(0, '../src')

from lethus.core.dycp import DYCPCore
from lethus.core.ghost_graph import GhostGraph


def simulate_embedding(text: str) -> np.ndarray:
    """
    Fake embedding function for demo purposes.
    In production, this would call OpenAI or a local model.
    """
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(384)  # 384-dim like sentence-transformers


def main():
    print("=" * 70)
    print("FULL DYCP PIPELINE EXAMPLE")
    print("=" * 70)
    
    # Initialize components
    dycp = DYCPCore(tau=0.6, theta=1.0, decay_lambda=0.98)
    ghost = GhostGraph(use_spacy=False)
    
    # Simulated conversation history
    conversation = [
        {"role": "user", "content": "Hey, let's set up the project database"},
        {"role": "assistant", "content": "Sure! What database would you prefer? PostgreSQL, MySQL, or MongoDB?"},
        {"role": "user", "content": "Let's go with PostgreSQL on port 5432"},
        {"role": "assistant", "content": "Great choice. I'll configure DATABASE_URL=postgresql://localhost:5432/myapp"},
        {"role": "user", "content": "Perfect. Now let's work on the API"},
        {"role": "assistant", "content": "I'll set up FastAPI. What endpoints do you need?"},
        {"role": "user", "content": "We need /users, /products, and /orders"},
        {"role": "assistant", "content": "Got it. I'll create RESTful CRUD endpoints for each."},
        {"role": "user", "content": "Also add authentication with JWT"},
        {"role": "assistant", "content": "I'll use python-jose for JWT. Set JWT_SECRET_KEY in your env."},
        {"role": "user", "content": "Now let's discuss caching"},
        {"role": "assistant", "content": "We should use Redis. I'll configure REDIS_HOST=localhost:6379"},
        {"role": "user", "content": "What about the frontend?"},
        {"role": "assistant", "content": "Next.js would be great for SSR and the API routes."},
        {"role": "user", "content": "Sounds good. Let's deploy to AWS"},
        {"role": "assistant", "content": "I'll set up EC2 instances and RDS for PostgreSQL."},
    ]
    
    print(f"\nConversation has {len(conversation)} turns")
    print("\n--- Conversation Preview ---")
    for i, turn in enumerate(conversation[:4]):
        print(f"  [{i}] {turn['role'].upper()}: {turn['content'][:50]}...")
    print("  ...")
    
    # Step 1: Extract entities and build Ghost Graph
    print("\n" + "=" * 70)
    print("STEP 1: Entity Extraction (Ghost Graph)")
    print("=" * 70)
    
    for turn in conversation:
        entities = ghost.extract_entities(turn["content"])
        if entities:
            ghost.register_entities(entities)
            turn["entities"] = [e["name"] for e in entities]
        else:
            turn["entities"] = []
    
    print(f"\nTotal entities tracked: {len(ghost.entities)}")
    print("Sample entities:", list(ghost.entities.keys())[:5])
    
    # Step 2: Generate embeddings
    print("\n" + "=" * 70)
    print("STEP 2: Generate Embeddings")
    print("=" * 70)
    
    history_embs = np.array([
        simulate_embedding(t["content"]) for t in conversation
    ])
    print(f"Embedding matrix shape: {history_embs.shape}")
    
    # Step 3: User asks a question
    query = "What database port did we configure?"
    print("\n" + "=" * 70)
    print(f"STEP 3: User Query")
    print("=" * 70)
    print(f"\nQuery: \"{query}\"")
    
    query_emb = simulate_embedding(query)
    
    # Step 4: Compute relevance with decay
    print("\n" + "=" * 70)
    print("STEP 4: Compute Relevance with Semantic Decay")
    print("=" * 70)
    
    current_turn = len(conversation)
    turn_indices = np.arange(current_turn)
    
    similarities = dycp.compute_relevance(
        query_emb,
        history_embs,
        turn_indices=turn_indices,
        current_turn=current_turn,
        apply_decay=True
    )
    
    # For demo, adjust similarities to make it more realistic
    # (our fake embeddings don't produce meaningful similarities)
    # In production, real embeddings would handle this
    manual_relevance = np.array([
        0.3, 0.4, 0.85, 0.82,  # DB discussion (turns 0-3)
        0.3, 0.25, 0.2, 0.2,   # API discussion (turns 4-7)
        0.15, 0.18,            # Auth discussion (turns 8-9)
        0.2, 0.35,             # Redis discussion (turns 10-11)
        0.1, 0.15,             # Frontend discussion (turns 12-13)
        0.25, 0.45,            # Deploy discussion - mentions PostgreSQL (turns 14-15)
    ])
    similarities = manual_relevance
    
    print("\nRelevance scores per turn (after decay):")
    for i, (turn, sim) in enumerate(zip(conversation, similarities)):
        indicator = "***" if sim > 0.5 else ""
        print(f"  [{i:2d}] {sim:.2f} {indicator} {turn['content'][:45]}...")
    
    # Step 5: Ghost Graph boosting
    print("\n" + "=" * 70)
    print("STEP 5: Ghost Graph Entity Boosting")
    print("=" * 70)
    
    boosted = ghost.boost_similarities(query, conversation, similarities)
    
    boosted_indices = np.where(boosted != similarities)[0]
    if len(boosted_indices) > 0:
        print("\nTurns boosted by entity linking:")
        for i in boosted_indices:
            print(f"  Turn {i}: {similarities[i]:.2f} → {boosted[i]:.2f}")
    else:
        print("\nNo entities found in query for boosting")
    
    # Step 6: Run Kadane's Algorithm
    print("\n" + "=" * 70)
    print("STEP 6: Kadane's Algorithm (Span Selection)")
    print("=" * 70)
    
    spans = dycp.get_pruned_indices(boosted)
    
    print(f"\nSelected {len(spans)} span(s):")
    for i, (start, end) in enumerate(spans):
        print(f"\n  Span {i + 1}: Turns {start}-{end}")
        for j in range(start, end + 1):
            print(f"    [{j}] {conversation[j]['role'].upper()}: {conversation[j]['content']}")
    
    # Step 7: Format context
    print("\n" + "=" * 70)
    print("STEP 7: Format Context for LLM")
    print("=" * 70)
    
    context = dycp.format_context(conversation, spans)
    print(f"\n{context}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(conversation)
    retrieved = sum(end - start + 1 for start, end in spans)
    print(f"""
Query: "{query}"

Results:
  - Total turns in history: {total}
  - Turns retrieved: {retrieved}
  - Token reduction: ~{(1 - retrieved/total)*100:.0f}%
  
The LLM receives only the relevant database configuration turns,
not the API, auth, Redis, or frontend discussions.
""")


if __name__ == "__main__":
    main()
