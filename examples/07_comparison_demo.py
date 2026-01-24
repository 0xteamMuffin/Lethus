"""
Example 7: DYCP vs Full Context Comparison
Demonstrates why DYCP outperforms sending all context.
"""
import numpy as np


def main():
    print("=" * 70)
    print("DYCP vs FULL CONTEXT: A Side-by-Side Comparison")
    print("=" * 70)
    
    # Simulated long conversation
    conversation = [
        # Topic 1: Setup (turns 0-9)
        "Let's build a REST API",
        "I'll use FastAPI with Python",
        "We should add authentication",
        "JWT would work well",
        "I'll store users in PostgreSQL",
        "The DB is on localhost:5432",  # <-- Relevant to DB query
        "Password hashing with bcrypt",
        "Add rate limiting too",
        "100 requests per minute",
        "Sounds good",
        
        # Topic 2: Frontend (turns 10-19) - NOISE for DB query
        "Now let's do the frontend",
        "React or Vue?",
        "Let's use React with TypeScript",
        "I'll set up Vite for bundling",
        "Add Tailwind for styling",
        "Should we use Redux?",
        "Zustand is simpler",
        "Good call, I'll add that",
        "Don't forget testing",
        "Jest and React Testing Library",
        
        # Topic 3: Deployment (turns 20-29) - Some relevant, mostly noise
        "Time to deploy",
        "Docker containers?",
        "Yes, with docker-compose",
        "I'll create the Dockerfile",
        "PostgreSQL in a container too",  # <-- Slightly relevant
        "Or use AWS RDS?",
        "RDS is easier to manage",
        "I'll configure the RDS instance",
        "Don't forget security groups",
        "And SSL certificates",
        
        # Topic 4: Current (turn 30+)
        "Actually, what port is the database on again?",  # User's question
    ]
    
    num_turns = len(conversation)
    user_query = conversation[-1]
    
    print(f"\nConversation length: {num_turns} turns")
    print(f"User's query: \"{user_query}\"")
    
    # Simulate relevance scores (in reality, these come from embeddings)
    relevance_scores = [
        0.3, 0.3, 0.35, 0.3, 0.6, 0.92,  # Turn 5 is highly relevant (port 5432)
        0.4, 0.25, 0.2, 0.15,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # Frontend - irrelevant
        0.2, 0.35, 0.3, 0.25, 0.55,  # Turn 24 mentions PostgreSQL
        0.3, 0.25, 0.2, 0.15, 0.1,
        1.0,  # Current query (always relevant to itself)
    ]
    
    print("\n" + "=" * 70)
    print("APPROACH 1: FULL CONTEXT")
    print("=" * 70)
    
    full_context_tokens = sum(len(t.split()) * 1.3 for t in conversation)  # Rough token estimate
    print(f"""
What happens:
  - Send ALL {num_turns} turns to the LLM
  - ~{int(full_context_tokens)} tokens of input context
  - LLM must find relevant info in a sea of noise
  
Problems:
  - Attention dilution: Model pays attention to frontend discussion
  - Potential confusion: "PostgreSQL in a container" vs "RDS" conflict
  - Higher latency: More tokens = slower response
  - Higher cost: $0.01 per 1K tokens × {int(full_context_tokens/1000*100)/100} = ${full_context_tokens/1000 * 0.01:.4f}
  
Typical failure mode:
  "Based on our discussion about deployment, it seems you're considering 
  AWS RDS for PostgreSQL, but I don't see a specific port mentioned."
  
  (Model got confused by the later RDS discussion)
""")
    
    print("=" * 70)
    print("APPROACH 2: DYCP (Our Method)")
    print("=" * 70)
    
    # Simulate DYCP span selection
    # High relevance turns: 4, 5, 6 (the DB setup discussion)
    selected_turns = [4, 5, 6]  # "PostgreSQL", "localhost:5432", "bcrypt"
    
    selected_context = [conversation[i] for i in selected_turns]
    dycp_tokens = sum(len(t.split()) * 1.3 for t in selected_context)
    
    print(f"""
What happens:
  1. Compute semantic similarity for all turns
  2. Apply decay (older turns get penalized unless very relevant)
  3. Run Kadane's Algorithm to find coherent span
  
Selected span: Turns {selected_turns[0]}-{selected_turns[-1]}
""")
    
    for i in selected_turns:
        score = relevance_scores[i]
        print(f"  [{i}] (score={score:.2f}) \"{conversation[i]}\"")
    
    print(f"""
Benefits:
  - Only {len(selected_turns)} turns sent (vs {num_turns})
  - ~{int(dycp_tokens)} tokens (vs {int(full_context_tokens)}) = {(1-dycp_tokens/full_context_tokens)*100:.0f}% reduction
  - No noise from frontend or deployment discussions
  - Coherent context: The DB setup conversation flows naturally
  
LLM Response:
  "The PostgreSQL database is configured on localhost:5432, as we 
  discussed during the initial API setup."
  
  (Correct, confident, no confusion)
""")
    
    print("=" * 70)
    print("SIDE-BY-SIDE METRICS")
    print("=" * 70)
    
    metrics = [
        ("Input Tokens", f"~{int(full_context_tokens)}", f"~{int(dycp_tokens)}", f"{(1-dycp_tokens/full_context_tokens)*100:.0f}% less"),
        ("Relevant Turns", "31 (all)", "3 (selected)", "10x reduction"),
        ("Noise Ratio", "~80%", "~0%", "Eliminated"),
        ("Latency (est.)", "2.5s", "1.1s", "2.3x faster"),
        ("Cost per Query", f"${full_context_tokens/1000 * 0.01:.4f}", f"${dycp_tokens/1000 * 0.01:.5f}", f"{full_context_tokens/dycp_tokens:.0f}x cheaper"),
        ("Accuracy (research)", "75.13%", "83.27%", "+10.8%"),
    ]
    
    print(f"\n{'Metric':<20} {'Full Context':<18} {'DYCP':<18} {'Improvement'}")
    print("-" * 75)
    for metric, full, dycp, improvement in metrics:
        print(f"{metric:<20} {full:<18} {dycp:<18} {improvement}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
DYCP wins not just on efficiency, but on ACCURACY.

Why? Because irrelevant context is ACTIVELY HARMFUL:
  - It dilutes attention away from relevant content
  - It can introduce conflicting information
  - It increases the chance of hallucination

Less context (when selected well) = Better answers.
""")


if __name__ == "__main__":
    main()
