"""FastAPI Application Server for BMW M Production RAG Chatbot v2.

Exposes REST API endpoints for chat generation, provenance verification,
operational telemetry metrics, and static asset serving.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from generation.generator import generate_rag_response
from generation.schemas import ChatRequest, RAGResponse
from retrieval.index import global_vector_index
from telemetry.storage import get_aggregated_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes vector store index on server startup."""
    logger.info("Initializing BMW M RAG Vector Index...")
    global_vector_index.build_or_load()
    logger.info("BMW M Production RAG System Online.")
    yield


app = FastAPI(
    title="BMW M Production RAG System",
    description="Enterprise-grade RAG chatbot with strict provenance and deterministic guardrails.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for flexible deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=RAGResponse, summary="Query RAG Pipeline")
async def chat_endpoint(request: ChatRequest) -> RAGResponse:
    """Processes user query through the complete RAG and guardrails pipeline."""
    try:
        response = generate_rag_response(query=request.query)
        return response
    except Exception as e:
        logger.error(f"Chat pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", summary="Retrieve Operational Telemetry Metrics")
async def metrics_endpoint():
    """Returns aggregated latency percentiles, token usage, and cost analytics."""
    return get_aggregated_metrics()


@app.get("/api/health", summary="System Health Check")
async def health_endpoint():
    """Returns service health status and vector store chunk count."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "indexed_chunks": len(global_vector_index.chunks),
        "similarity_threshold": settings.SIMILARITY_THRESHOLD
    }


# Mount Static UI Files
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=True)
