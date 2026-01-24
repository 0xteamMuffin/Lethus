import argparse


def run_server():
    import uvicorn
    from .config import settings
    
    print("=" * 60)
    print("  LETHUS - DYCP Context Reduction Proxy")
    print("=" * 60)
    print(f"  Proxy endpoint: http://{settings.api_host}:{settings.api_port}/v1/chat/completions")
    print(f"  REST API:       http://{settings.api_host}:{settings.api_port}/api/")
    print("=" * 60)
    
    uvicorn.run(
        "lethus.api.rest:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(description="Lethus - DYCP Context Reduction Proxy")
    
    parser.add_argument("--init-db", action="store_true", help="Initialize database tables")
    parser.add_argument("--host", type=str, default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    
    args = parser.parse_args()
    
    if args.init_db:
        from .storage.postgres import init_db
        print("Initializing database...")
        init_db()
        print("Database initialized successfully.")
        return
    
    if args.host or args.port:
        from .config import settings
        if args.host:
            settings.api_host = args.host
        if args.port:
            settings.api_port = args.port
    
    run_server()


if __name__ == "__main__":
    main()
