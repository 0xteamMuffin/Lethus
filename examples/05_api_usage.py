"""
Example 5: REST API Usage
Shows how to interact with Lethus via HTTP endpoints.

Prerequisites:
  1. cd docker && docker-compose up -d
  2. lethus --mode=api
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def example_conversation():
    """Simulate a conversation and test memory retrieval."""
    
    print("=" * 60)
    print("LETHUS REST API EXAMPLE")
    print("=" * 60)
    print("\nNote: Make sure Lethus is running with: lethus --mode=api\n")
    
    # Example conversation to store
    messages = [
        ("user", "I want to build a Python web scraper"),
        ("assistant", "I'll help you build a web scraper. We'll use BeautifulSoup and requests."),
        ("user", "Great, let's scrape news headlines from BBC"),
        ("assistant", "Here's the code to scrape BBC headlines using CSS selectors..."),
        ("user", "Now I want to store them in a database"),
        ("assistant", "I'll add SQLite storage. The database file will be at ./headlines.db"),
        ("user", "Perfect. Can we also add scheduling?"),
        ("assistant", "I'll use APScheduler to run the scraper every hour."),
        ("user", "What about error handling?"),
        ("assistant", "I'll add try-except blocks and logging to ./scraper.log"),
    ]
    
    print("--- Storing Conversation ---\n")
    
    # This would be the API call format (shown as example)
    print("Example API calls to store messages:\n")
    for role, content in messages[:3]:
        payload = {
            "user_id": "demo_user",
            "role": role,
            "content": content
        }
        print(f"POST {BASE_URL}/api/store")
        print(f"Body: {json.dumps(payload, indent=2)}\n")
    
    print("...(8 more messages)\n")
    
    # Query example
    print("=" * 60)
    print("QUERY EXAMPLE")
    print("=" * 60)
    
    query_payload = {
        "user_id": "demo_user",
        "message": "Where is the database file stored?",
        "openai_api_key": "sk-..."  # Your API key
    }
    
    print(f"\nPOST {BASE_URL}/api/chat")
    print(f"Body: {json.dumps(query_payload, indent=2)}")
    
    print("\n--- Expected Response ---\n")
    expected_response = {
        "conversation_id": 1,
        "turn_id": 11,
        "message": "Based on our previous discussion, the SQLite database file is stored at ./headlines.db",
        "retrieved_context": {
            "spans": [
                {"start_index": 4, "end_index": 5, "relevance_score": 0.87}
            ],
            "confidence": {"confident": True, "score": 0.85}
        }
    }
    print(json.dumps(expected_response, indent=2))
    
    print("\n" + "=" * 60)
    print("AVAILABLE ENDPOINTS")
    print("=" * 60)
    
    endpoints = [
        ("POST", "/api/chat", "Send message, get memory-enhanced response"),
        ("POST", "/api/conversations", "Create new conversation"),
        ("GET", "/api/conversations/{id}", "Get conversation details"),
        ("GET", "/api/conversations/{id}/turns", "Get conversation history"),
        ("DELETE", "/api/conversations/{id}", "Delete conversation"),
        ("GET", "/api/stats", "Memory system statistics"),
        ("GET", "/api/entity-graph", "View Ghost Graph state"),
        ("POST", "/api/clear", "Clear all memory"),
    ]
    
    print(f"\n{'Method':<8} {'Endpoint':<30} {'Description'}")
    print("-" * 70)
    for method, endpoint, desc in endpoints:
        print(f"{method:<8} {endpoint:<30} {desc}")
    
    print(f"\nFull docs: {BASE_URL}/docs (Swagger UI)")
    

def test_live_api():
    """Actually test the API if it's running."""
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=2)
        if response.ok:
            print("\n" + "=" * 60)
            print("LIVE API TEST")
            print("=" * 60)
            print("\nAPI is running! Stats:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\nAPI returned error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("\n(API not running - examples shown are for reference)")
    except Exception as e:
        print(f"\n(Could not connect to API: {e})")


if __name__ == "__main__":
    example_conversation()
    test_live_api()
