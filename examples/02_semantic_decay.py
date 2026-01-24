"""
Example 2: Semantic Decay
Shows how older messages need higher base relevance to be recalled.
"""
import numpy as np
import sys
sys.path.insert(0, '../src')

from lethus.core.dycp import DYCPCore


def main():
    dycp = DYCPCore(decay_lambda=0.98)
    
    print("=" * 60)
    print("SEMANTIC DECAY EXAMPLE")
    print("=" * 60)
    print("\nFormula: decayed_score = similarity * (0.98 ^ age)")
    print("This means older messages need HIGHER base similarity to surface.\n")
    
    # Simulate embeddings (simplified as 1D for demo)
    query_emb = np.array([1.0, 0.0, 0.0, 0.5])  # Query embedding
    
    # Two messages with SAME base similarity but different ages
    message_old_emb = np.array([0.9, 0.1, 0.0, 0.4])   # From 100 turns ago
    message_new_emb = np.array([0.9, 0.1, 0.0, 0.4])   # From 5 turns ago
    
    history_embs = np.array([message_old_emb, message_new_emb])
    turn_indices = np.array([0, 95])  # Turn 0 and Turn 95
    current_turn = 100
    
    # Without decay
    raw_similarities = dycp.compute_relevance(
        query_emb, history_embs, apply_decay=False
    )
    
    # With decay
    decayed_similarities = dycp.compute_relevance(
        query_emb, history_embs, 
        turn_indices=turn_indices,
        current_turn=current_turn,
        apply_decay=True
    )
    
    print("Scenario: Two messages with IDENTICAL content and similarity")
    print("  - Message A: From 100 turns ago (age = 100)")
    print("  - Message B: From 5 turns ago (age = 5)")
    
    print(f"\n{'Without Decay:':<20} Both have same score")
    print(f"  Message A (old): {raw_similarities[0]:.4f}")
    print(f"  Message B (new): {raw_similarities[1]:.4f}")
    
    print(f"\n{'With Decay (λ=0.98):':<20}")
    print(f"  Message A (old): {decayed_similarities[0]:.4f}  (× 0.98^100 = × {0.98**100:.4f})")
    print(f"  Message B (new): {decayed_similarities[1]:.4f}  (× 0.98^5 = × {0.98**5:.4f})")
    
    print("\n" + "=" * 60)
    print("DECAY CURVE")
    print("=" * 60)
    ages = [0, 10, 25, 50, 100, 200, 500]
    print(f"\n{'Age (turns)':<15} {'Decay Factor':<15} {'Effect'}")
    print("-" * 50)
    for age in ages:
        factor = 0.98 ** age
        effect = "Full weight" if factor > 0.8 else "Moderate" if factor > 0.3 else "Heavily penalized"
        print(f"{age:<15} {factor:<15.4f} {effect}")
    
    print("\n" + "=" * 60)
    print("PRACTICAL IMPLICATION")
    print("=" * 60)
    print("""
If you discussed an API key 200 turns ago with similarity 0.8,
its decayed score is: 0.8 × 0.98^200 = 0.8 × 0.018 = 0.014

But if you mentioned it again 10 turns ago with similarity 0.7,
its decayed score is: 0.7 × 0.98^10 = 0.7 × 0.817 = 0.572

The recent mention surfaces, old one doesn't (unless query is very specific).
""")


if __name__ == "__main__":
    main()
