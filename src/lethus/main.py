"""
Lethus - Main Entry Point

Usage:
    lethus                # Run the server (default)
    lethus --init-db      # Initialize database
    
Or via Python:
    python -m lethus.main

The server exposes:
    /v1/chat/completions  - OpenAI-compatible proxy with DYCP context reduction
    /api/*                - REST API for direct integration
"""
import argparse


def run_server():
    """Run the Lethus server with proxy and REST API."""
    import uvicorn
    from .config import settings
    
    print("=" * 60)
    print("  LETHUS - DYCP Context Reduction Proxy")
    print("=" * 60)
    print(f"  Proxy endpoint: http://{settings.api_host}:{settings.api_port}/v1/chat/completions")
    print(f"  REST API:       http://{settings.api_host}:{settings.api_port}/api/")
    print("=" * 60)
    print()
    print("  To use with any OpenAI-compatible client:")
    print(f'    base_url = "http://{settings.api_host}:{settings.api_port}/v1"')
    print()
    
    uvicorn.run(
        "lethus.api.rest:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Lethus - DYCP Context Reduction Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lethus is an OpenAI-compatible proxy that applies Dynamic Context Pruning
to reduce conversation history to only relevant spans.

Usage with any OpenAI client:
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="your-openai-key"
    )
    
The proxy automatically:
    1. Intercepts chat requests
    2. Applies DYCP (Kadane's Algorithm) to select relevant history
    3. Forwards reduced context to OpenAI
    4. Stores interactions for future retrieval
        """
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize PostgreSQL database tables and exit"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override host (default: from config)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port (default: from config)"
    )
    
    args = parser.parse_args()
    
    if args.init_db:
        from .storage.postgres import init_db
        print("Initializing database...")
        init_db()
        print("Database initialized successfully.")
        return
    
    # Override settings if provided
    if args.host or args.port:
        from .config import settings
        if args.host:
            settings.api_host = args.host
        if args.port:
            settings.api_port = args.port
    
    run_server()


if __name__ == "__main__":
    main()
