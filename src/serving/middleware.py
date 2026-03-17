"""
Prometheus metrics middleware for FastAPI.
Tracks request latency, recommendation latency, candidate counts, and cache statistics.
"""
import time
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, Summary
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ------------------------------------------------------------------ #
# Prometheus Metric Definitions                                        #
# ------------------------------------------------------------------ #

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint", "status_code"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
)

RECOMMENDATION_LATENCY_MS = Histogram(
    "recommendation_latency_ms",
    "End-to-end recommendation pipeline latency in milliseconds",
    buckets=[5, 10, 25, 50, 75, 100, 200, 500, 1000, 2000, 5000],
)

RETRIEVAL_LATENCY_MS = Histogram(
    "retrieval_latency_ms",
    "FAISS ANN retrieval latency in milliseconds",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200],
)

RANKING_LATENCY_MS = Histogram(
    "ranking_latency_ms",
    "LightGBM re-ranking latency in milliseconds",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200],
)

CANDIDATES_RETRIEVED = Gauge(
    "candidates_retrieved_total",
    "Number of ANN candidates retrieved per request",
)

CACHE_HITS = Counter(
    "recommendation_cache_hits_total",
    "Total number of recommendation cache hits",
)

CACHE_MISSES = Counter(
    "recommendation_cache_misses_total",
    "Total number of recommendation cache misses",
)

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of currently active HTTP requests",
)

RECOMMENDATION_ERRORS = Counter(
    "recommendation_errors_total",
    "Total number of recommendation errors",
    ["error_type"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that records Prometheus metrics for every HTTP request.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        endpoint = self._get_endpoint_label(request)
        method = request.method

        ACTIVE_REQUESTS.inc()
        t_start = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception as exc:
            ACTIVE_REQUESTS.dec()
            RECOMMENDATION_ERRORS.labels(error_type=type(exc).__name__).inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()

        latency = time.perf_counter() - t_start

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).observe(latency)

        REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        return response

    @staticmethod
    def _get_endpoint_label(request: Request) -> str:
        """Normalize path to avoid high-cardinality labels from path params."""
        path = request.url.path
        # Normalize /recommend to avoid user-id cardinality issues
        if path.startswith("/recommend"):
            return "/recommend"
        if path.startswith("/health"):
            return "/health"
        if path.startswith("/metrics"):
            return "/metrics"
        if path.startswith("/model"):
            return "/model/info"
        return path


def record_recommendation_metrics(
    latency_ms: float,
    retrieval_ms: float,
    ranking_ms: float,
    n_candidates: int,
    cache_hit: bool,
) -> None:
    """
    Record fine-grained recommendation pipeline metrics.
    Called from the recommendation endpoint handler.
    """
    RECOMMENDATION_LATENCY_MS.observe(latency_ms)
    RETRIEVAL_LATENCY_MS.observe(retrieval_ms)
    RANKING_LATENCY_MS.observe(ranking_ms)
    CANDIDATES_RETRIEVED.set(n_candidates)

    if cache_hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()
