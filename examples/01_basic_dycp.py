"""
Example 1: Basic DYCP Algorithm
Demonstrates how Kadane's Algorithm selects relevant conversation spans.
"""
import numpy as np
import sys
sys.path.insert(0, '../src')

from lethus.core.dycp import DYCPCore


def main():
    # Initialize DYCP with default parameters
    dycp = DYCPCore(tau=0.6, theta=1.0, decay_lambda=0.98)
    
    # Simulate a conversation history with relevance scores
    # Imagine user asked: "What database did we configure?"
    # These are cosine similarity scores for each turn
    similarities = np.array([
        0.2,   # Turn 0: "Hello, how are you?"
        0.15,  # Turn 1: "I'm good, let's start coding"
        0.3,   # Turn 2: "What stack should we use?"
        0.85,  # Turn 3: "Let's use PostgreSQL on port 5432"  <-- RELEVANT
        0.78,  # Turn 4: "Good choice, I'll set up the schema" <-- RELEVANT
        0.72,  # Turn 5: "Here's the connection string..."    <-- RELEVANT
        0.25,  # Turn 6: "Now let's work on the API"
        0.18,  # Turn 7: "I'll use FastAPI"
        0.22,  # Turn 8: "Sounds good"
        0.65,  # Turn 9: "Remember the DB port is 5432"       <-- RELEVANT (later mention)
        0.3,   # Turn 10: "Got it, thanks"
    ])
    
    print("=" * 60)
    print("DYCP EXAMPLE: Finding Relevant Conversation Spans")
    print("=" * 60)
    print(f"\nQuery: 'What database did we configure?'")
    print(f"\nSimilarity scores per turn:")
    for i, s in enumerate(similarities):
        relevance = "HIGH" if s > 0.6 else "MED" if s > 0.4 else "LOW"
        print(f"  Turn {i:2d}: {s:.2f} [{relevance}]")
    
    # Run Kadane's Algorithm
    spans = dycp.get_pruned_indices(similarities)
    
    print(f"\n{'=' * 60}")
    print("SELECTED SPANS (using Kadane's Algorithm)")
    print("=" * 60)
    
    for i, (start, end) in enumerate(spans):
        print(f"\nSpan {i + 1}: Turns {start} to {end}")
        print(f"  Content: Turns about database configuration")
        span_scores = similarities[start:end + 1]
        print(f"  Scores: {span_scores}")
        print(f"  Avg relevance: {np.mean(span_scores):.2f}")
    
    # Show what we would have retrieved vs full context
    total_turns = len(similarities)
    retrieved_turns = sum(end - start + 1 for start, end in spans)
    
    print(f"\n{'=' * 60}")
    print("EFFICIENCY")
    print("=" * 60)
    print(f"Total turns in history: {total_turns}")
    print(f"Turns retrieved by DYCP: {retrieved_turns}")
    print(f"Reduction: {(1 - retrieved_turns / total_turns) * 100:.1f}%")
    

if __name__ == "__main__":
    main()
