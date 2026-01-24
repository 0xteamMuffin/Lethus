from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..config import settings
from ..storage.postgres import init_db
from .proxy import router as proxy_router
from .rest_routes import router as rest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Lethus Proxy",
    description="OpenAI-compatible proxy with DYCP context reduction",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Lethus-Enhanced-Mode",
        "X-Lethus-Original-Messages",
        "X-Lethus-Reduced-Messages",
        "X-Lethus-Original-Tokens",
        "X-Lethus-Reduced-Tokens",
        "X-Lethus-Tokens-Saved",
        "X-Lethus-Reduction-Percent",
        "X-Lethus-Spans-Found",
        "X-Lethus-Processing-Ms",
        "X-Lethus-Ghost-Entities",
        "X-Lethus-Ghost-Boosts",
        "X-Lethus-Decay-Lambda",
        "X-Lethus-Tau",
        "X-Lethus-Theta",
        "X-Lethus-Entity-Names",
        "X-Lethus-Span-Details",
        "X-Lethus-Boost-Count",
    ],
)

app.include_router(proxy_router, prefix="/v1", tags=["OpenAI API"])

app.include_router(rest_router, prefix="/api", tags=["REST API"])


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Lethus Proxy",
        "version": "0.1.0"
    }


def main():
    import uvicorn
    uvicorn.run(
        "lethus.api.rest:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
