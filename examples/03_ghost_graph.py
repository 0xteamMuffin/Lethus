"""
Example 3: Ghost Graph Entity Extraction and Linking
Demonstrates how we track entities and resolve pronouns.
"""
import sys
sys.path.insert(0, '../src')

from lethus.core.ghost_graph import GhostGraph


def main():
    # Initialize Ghost Graph (without spaCy for this demo)
    ghost = GhostGraph(use_spacy=True)
    
    print("=" * 60)
    print("GHOST GRAPH EXAMPLE: Entity Extraction & Linking")
    print("=" * 60)
    
    # Simulate conversation turns
    turns = [
        "Let's configure the AWS_ACCESS_KEY for our deployment",
        "I'll also need the DATABASE_URL and REDIS_HOST",
        "The API endpoint is https://api.example.com/v2",
        "John said we should use port 5432 for PostgreSQL",
        "Contact support@company.com if there are issues",
        "The server IP is 192.168.1.100",
    ]
    
    print("\n--- Processing Conversation Turns ---\n")
    
    for i, turn in enumerate(turns):
        print(f"Turn {i}: \"{turn}\"")
        entities = ghost.extract_entities(turn)
        
        if entities:
            print(f"  Extracted: {[e['name'] + ' (' + e['type'] + ')' for e in entities]}")
            ghost.register_entities(entities)
        else:
            print("  Extracted: (none)")
        print()
    
    print("=" * 60)
    print("GHOST GRAPH STATE")
    print("=" * 60)
    print(ghost.get_state_summary())
    
    # Demonstrate entity linking
    print("\n" + "=" * 60)
    print("ENTITY LINKING DEMO")
    print("=" * 60)
    
    entity = "AWS_ACCESS_KEY"
    linked = ghost.get_linked_entities(entity, depth=1)
    print(f"\nEntities linked to '{entity}':")
    print(f"  {linked}")
    print(f"\n  (These entities appeared in the same turn)")
    
    # Demonstrate finding mentions
    print("\n" + "=" * 60)
    print("FINDING ENTITY MENTIONS")
    print("=" * 60)
    
    queries = [
        "What was that AWS key again?",
        "Tell me about the database connection",
        "What IP should I use?",
    ]
    
    for query in queries:
        mentions = ghost.find_entity_mentions(query)
        print(f"\nQuery: \"{query}\"")
        print(f"  Found entities: {mentions if mentions else '(none - would use semantic search)'}")
    
    # Demonstrate boosting
    print("\n" + "=" * 60)
    print("SIMILARITY BOOSTING")
    print("=" * 60)
    print("""
When a query mentions a known entity, we boost similarity scores
for turns containing linked entities.

Example:
  Query: "What about the AWS setup?"
  - Turn 0 mentions AWS_ACCESS_KEY → boost by 1.2x
  - Turn 1 mentions DATABASE_URL (linked to AWS_ACCESS_KEY) → boost by 1.2x
  - Other turns → no boost

This helps surface related context even without exact keyword matches.
""")


if __name__ == "__main__":
    main()
