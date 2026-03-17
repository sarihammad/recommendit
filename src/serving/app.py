"""
FastAPI application for the RecommendIt recommendation service.
Exposes endpoints for health checks, recommendations, model info, and Prometheus metrics.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.config import settings
from src.serving.middleware import PrometheusMiddleware, record_recommendation_metrics
from src.serving.recommender import RecommendationPipeline, RecommendationResult

logger = logging.getLogger(__name__)

# Global pipeline instance (initialized at startup)
_pipeline: Optional[RecommendationPipeline] = None


# ------------------------------------------------------------------ #
# Request / Response Models                                           #
# ------------------------------------------------------------------ #

class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for", gt=0)
    k: int = Field(default=20, description="Number of recommendations to return", ge=1, le=100)
    use_cache: bool = Field(default=True, description="Whether to use Redis recommendation cache")


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    score: float
    rank: int
    retrieval_score: float
    genres: List[str]


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    latency_ms: float
    cache_hit: bool
    n_candidates: int


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    feature_store_backend: str
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_version: str
    embedding_dim: int
    n_users: int
    n_items: int
    index_stats: Dict[str, Any]
    ranker_info: Dict[str, Any]
    pipeline_stats: Dict[str, Any]


# ------------------------------------------------------------------ #
# Application Lifecycle                                               #
# ------------------------------------------------------------------ #

_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    global _pipeline
    logger.info("Starting RecommendIt API service...")
    try:
        _pipeline = RecommendationPipeline()
        _pipeline.load()
        logger.info("Pipeline ready.")
    except Exception as exc:
        logger.error("Failed to load pipeline: %s", exc, exc_info=True)
        # Allow startup to proceed even if models are missing (useful for health checks)
        _pipeline = None
    yield
    logger.info("Shutting down RecommendIt API service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RecommendIt",
        description=(
            "Production-grade movie recommendation API using Two-Tower embeddings, "
            "FAISS ANN search, and LightGBM LambdaMART re-ranking."
        ),
        version=settings.MODEL_VERSION,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Routes                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/health", response_model=HealthResponse, tags=["Operations"])
    async def health_check():
        """Service health check. Returns pipeline readiness and uptime."""
        uptime = time.time() - _startup_time
        if _pipeline is not None and _pipeline._loaded:
            fs_stats = _pipeline.feature_store.stats()
            return HealthResponse(
                status="healthy",
                pipeline_loaded=True,
                feature_store_backend=fs_stats.get("backend", "unknown"),
                model_version=settings.MODEL_VERSION,
                uptime_seconds=round(uptime, 2),
            )
        return HealthResponse(
            status="degraded",
            pipeline_loaded=False,
            feature_store_backend="none",
            model_version=settings.MODEL_VERSION,
            uptime_seconds=round(uptime, 2),
        )

    @app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
    async def get_recommendations(request: RecommendRequest):
        """
        Generate personalized movie recommendations for a user.

        Returns top-K items ranked by the LightGBM LambdaMART ranker,
        with a fallback to popularity-based recommendations for unknown users.
        """
        if _pipeline is None or not _pipeline._loaded:
            raise HTTPException(status_code=503, detail="Recommendation pipeline not available")

        t_start = time.perf_counter()
        cache_hit = False

        # Check cache before hitting the full pipeline
        if request.use_cache:
            cached = _pipeline.feature_store.get_cached_recommendations(request.user_id)
            if cached is not None:
                cache_hit = True
                latency_ms = (time.perf_counter() - t_start) * 1000
                record_recommendation_metrics(
                    latency_ms=latency_ms,
                    retrieval_ms=0.0,
                    ranking_ms=0.0,
                    n_candidates=0,
                    cache_hit=True,
                )
                return RecommendResponse(
                    user_id=request.user_id,
                    recommendations=[RecommendationItem(**item) for item in cached[:request.k]],
                    latency_ms=round(latency_ms, 2),
                    cache_hit=True,
                    n_candidates=0,
                )

        try:
            results = _pipeline.get_recommendations(
                user_id=request.user_id,
                k=request.k,
                use_cache=False,  # We already checked above
            )
        except Exception as exc:
            logger.error("Recommendation error for user %d: %s", request.user_id, exc, exc_info=True)
            # Graceful degradation: return popularity-based recommendations
            results = _pipeline._popularity_recommendations(request.k)

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Record metrics
        stats = _pipeline.get_stats()
        record_recommendation_metrics(
            latency_ms=latency_ms,
            retrieval_ms=_pipeline.retrieval_latency.p50,
            ranking_ms=_pipeline.ranking_latency.p50,
            n_candidates=settings.TOP_K_CANDIDATES,
            cache_hit=False,
        )

        return RecommendResponse(
            user_id=request.user_id,
            recommendations=[
                RecommendationItem(
                    item_id=r.item_id,
                    title=r.title,
                    score=round(r.score, 6),
                    rank=r.rank,
                    retrieval_score=round(r.retrieval_score, 6),
                    genres=r.genres,
                )
                for r in results
            ],
            latency_ms=round(latency_ms, 2),
            cache_hit=cache_hit,
            n_candidates=settings.TOP_K_CANDIDATES,
        )

    @app.get("/metrics", response_class=PlainTextResponse, tags=["Operations"])
    async def prometheus_metrics():
        """Prometheus metrics endpoint in text format."""
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["Operations"])
    async def model_info():
        """Return model metadata, version, feature importance, and pipeline stats."""
        if _pipeline is None or not _pipeline._loaded:
            raise HTTPException(status_code=503, detail="Pipeline not loaded")

        return ModelInfoResponse(
            model_version=settings.MODEL_VERSION,
            embedding_dim=_pipeline.model.embed_dim,
            n_users=_pipeline.model.n_users,
            n_items=_pipeline.model.n_items,
            index_stats=_pipeline.faiss_index.stats(),
            ranker_info=_pipeline.ranker.model_info(),
            pipeline_stats=_pipeline.get_stats(),
        )

    @app.get("/items/{item_id}", tags=["Items"])
    async def get_item(item_id: int):
        """Retrieve metadata for a specific item by ID."""
        if _pipeline is None or not _pipeline._loaded:
            raise HTTPException(status_code=503, detail="Pipeline not loaded")
        title = _pipeline._item_titles.get(item_id)
        if title is None:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        return {
            "item_id": item_id,
            "title": title,
            "genres": _pipeline._item_genres.get(item_id, []),
        }

    return app


# Singleton app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
    uvicorn.run(
        "src.serving.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        workers=1,
    )
