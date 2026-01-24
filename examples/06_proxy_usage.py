"""
Example 06: Using Lethus as an OpenAI-Compatible Proxy

This demonstrates how to use Lethus as a drop-in replacement for OpenAI API.
The proxy automatically applies DYCP context reduction to your conversations.

Prerequisites:
    1. Start Lethus server: python -m lethus.main
    2. Have OpenAI API key set

Usage:
    python 06_proxy_usage.py
"""
from openai import OpenAI


def main():
    # Point to Lethus proxy instead of OpenAI directly
    client = OpenAI(
        base_url="http://localhost:8000/v1",  # Lethus proxy
        api_key="your-openai-api-key"  # Your real OpenAI key
    )
    
    print("=" * 60)
    print("Lethus Proxy Demo - DYCP Context Reduction")
    print("=" * 60)
    
    # Simulate a long conversation that would normally bloat context
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        # Old context (will be pruned if not relevant)
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity."},
        {"role": "user", "content": "How do I install packages?"},
        {"role": "assistant", "content": "Use pip: pip install package_name"},
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data."},
        {"role": "user", "content": "Tell me about the Eiffel Tower"},
        {"role": "assistant", "content": "The Eiffel Tower is a famous landmark in Paris, France."},
        # More relevant recent context
        {"role": "user", "content": "How do I use numpy arrays?"},
        {"role": "assistant", "content": "Import numpy as np, then use np.array([1, 2, 3]) to create arrays."},
        # Current query - DYCP will keep numpy-related history, prune weather/Eiffel
        {"role": "user", "content": "How do I reshape a numpy array?"},
    ]
    
    print(f"\nSending {len(messages)} messages to Lethus proxy...")
    print("DYCP will select only relevant spans for the current query.\n")
    
    # The proxy automatically:
    # 1. Receives the full message history
    # 2. Applies DYCP to select relevant spans
    # 3. Forwards reduced context to OpenAI
    # 4. Stores the interaction for future retrieval
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=False
    )
    
    print("Response:")
    print("-" * 40)
    print(response.choices[0].message.content)
    print("-" * 40)
    
    # Streaming example
    print("\n\nStreaming Example:")
    print("-" * 40)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Explain DYCP in one sentence."}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n" + "-" * 40)
    print("\nDone! Check server logs to see DYCP reduction stats.")


if __name__ == "__main__":
    main()
