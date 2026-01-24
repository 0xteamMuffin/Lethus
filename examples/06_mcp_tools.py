"""
Example 6: MCP Tools Reference
Shows what tools are available for LLM integration (Claude, etc.)

When Lethus runs as an MCP server, these tools become available
to any MCP-compatible LLM client.
"""


def main():
    print("=" * 70)
    print("LETHUS MCP TOOLS REFERENCE")
    print("=" * 70)
    print("""
When running Lethus as an MCP server (lethus --mode=mcp), the following
tools become available to LLM agents like Claude.
""")

    tools = [
        {
            "name": "store_interaction",
            "description": "Stores a conversation turn in long-term memory with entity extraction.",
            "args": {
                "role": "Either 'user' or 'assistant'",
                "content": "The full text content of the message"
            },
            "example": '''store_interaction(
    role="user",
    content="Let's use PostgreSQL on port 5432 for the database"
)
# Returns: "Stored in memory. Entities: ['PostgreSQL']"''',
            "when_to_use": "Call after EVERY user message and assistant response"
        },
        {
            "name": "get_context",
            "description": "Retrieves relevant conversation context using DYCP with Semantic Decay.",
            "args": {
                "query": "The user's current question or topic"
            },
            "example": '''get_context(query="What database port did we configure?")
# Returns:
# --- RELEVANT CONTEXT (DYCP + Decay + Ghost Graph) ---
# [USER]: Let's use PostgreSQL on port 5432 for the database
# [ASSISTANT]: Great, I'll configure DATABASE_URL with port 5432
# --- END CONTEXT ---''',
            "when_to_use": "Before answering questions that may reference past conversation"
        },
        {
            "name": "search_by_entity",
            "description": "Search memory for turns mentioning a specific entity.",
            "args": {
                "entity_name": "The entity to search for (e.g., 'API_KEY', 'John', 'AWS')"
            },
            "example": '''search_by_entity(entity_name="DATABASE_URL")
# Returns all turns mentioning DATABASE_URL''',
            "when_to_use": "When you know the exact name of a config, person, or concept"
        },
        {
            "name": "get_entity_graph",
            "description": "Returns the current state of the Ghost Graph (entity relationships).",
            "args": {},
            "example": '''get_entity_graph()
# Returns:
# --- GHOST GRAPH STATE ---
#   PostgreSQL (CONFIG) -> ['DATABASE_URL', 'port 5432']
#   AWS (ORG) -> ['EC2', 'S3', 'API_KEY']
# --- END GRAPH ---''',
            "when_to_use": "For debugging or understanding what entities are tracked"
        },
        {
            "name": "get_memory_stats",
            "description": "Returns statistics about the current memory state.",
            "args": {},
            "example": '''get_memory_stats()
# Returns:
# Memory Stats:
#   Total Turns: 156
#   Tracked Entities: 42
#   Prefetch Cache Size: 5
#   Decay Lambda: 0.98
#   DYCP Tau: 0.6
#   DYCP Theta: 1.0''',
            "when_to_use": "To check memory health or debug issues"
        },
        {
            "name": "clear_memory",
            "description": "Wipes all conversation history and resets the Ghost Graph.",
            "args": {},
            "example": '''clear_memory()
# Returns: "All memory cleared."''',
            "when_to_use": "When user wants to start fresh or for testing"
        },
    ]
    
    for tool in tools:
        print("\n" + "-" * 70)
        print(f"TOOL: {tool['name']}")
        print("-" * 70)
        print(f"\nDescription: {tool['description']}")
        print(f"\nWhen to use: {tool['when_to_use']}")
        
        if tool['args']:
            print("\nArguments:")
            for arg, desc in tool['args'].items():
                print(f"  - {arg}: {desc}")
        else:
            print("\nArguments: None")
        
        print(f"\nExample:\n{tool['example']}")
    
    print("\n" + "=" * 70)
    print("CLAUDE DESKTOP CONFIGURATION")
    print("=" * 70)
    print('''
To use Lethus with Claude Desktop, add to claude_desktop_config.json:

{
  "mcpServers": {
    "lethus": {
      "command": "lethus",
      "args": ["--mode=mcp"]
    }
  }
}

Then Claude can call these tools automatically during conversation.
''')

    print("=" * 70)
    print("TYPICAL WORKFLOW")
    print("=" * 70)
    print('''
1. User sends message
2. Claude calls store_interaction(role="user", content=message)
3. Claude calls get_context(query=message) to retrieve relevant history
4. Claude generates response using the retrieved context
5. Claude calls store_interaction(role="assistant", content=response)
6. Repeat!

The memory persists across sessions, so Claude remembers everything.
''')


if __name__ == "__main__":
    main()
