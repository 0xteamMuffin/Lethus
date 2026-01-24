"""
Lethus - Main Entry Point

Usage:
    lethus --mode=mcp     # Run MCP server only
    lethus --mode=api     # Run REST API only  
    lethus --mode=both    # Run both (default)
    
Or via Python:
    python -m lethus.main --mode=api
"""
import argparse
import sys
import asyncio
import threading


def run_mcp_server():
    """Run the MCP server"""
    from .api.mcp import main as mcp_main
    mcp_main()


def run_rest_api():
    """Run the REST API server"""
    from .api.rest import main as api_main
    api_main()


def run_both():
    """Run both MCP and REST API servers"""
    import uvicorn
    from .config import settings
    
    def start_api():
        uvicorn.run(
            "lethus.api.rest:app",
            host=settings.api_host,
            port=settings.api_port,
            log_level="info"
        )
    
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    print(f"REST API running on http://{settings.api_host}:{settings.api_port}")
    print("Starting MCP server...")
    
    # Run MCP in main thread (it uses stdio)
    run_mcp_server()


def main():
    parser = argparse.ArgumentParser(
        description="Lethus - DYCP Memory System with MCP + REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    lethus --mode=mcp     # Run MCP server for LLM integration
    lethus --mode=api     # Run REST API for web apps
    lethus --mode=both    # Run both servers
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["mcp", "api", "both"],
        default="mcp",
        help="Server mode: 'mcp' (MCP tools), 'api' (REST API), or 'both'"
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize PostgreSQL database tables and exit"
    )
    
    args = parser.parse_args()
    
    if args.init_db:
        from .storage.postgres import init_db
        print("Initializing database...")
        init_db()
        print("Database initialized successfully.")
        return
    
    if args.mode == "mcp":
        run_mcp_server()
    elif args.mode == "api":
        run_rest_api()
    else:
        run_both()


if __name__ == "__main__":
    main()
