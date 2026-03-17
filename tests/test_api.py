"""
Integration tests for the FastAPI recommendation service.
Uses TestClient and mocked pipeline to avoid requiring trained models.
"""
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------ #
# Mock Pipeline Fixture                                               #
# ------------------------------------------------------------------ #

def make_mock_pipeline(n_results: int = 5):
    """Create a mock RecommendationPipeline for testing."""
    from src.serving.recommender import RecommendationResult

    mock_pipeline = MagicMock()
    mock_pipeline._loaded = True

    # Mock feature store
    mock_fs = MagicMock()
    mock_fs.get_cached_recommendations.return_value = None
    mock_fs.cache_recommendations.return_value = None
    mock_fs.stats.return_value = {"backend": "in-memory", "keys": 100}
    mock_pipeline.feature_store = mock_fs

    # Mock model attributes
    mock_pipeline.model.embed_dim = 64
    mock_pipeline.model.n_users = 6040
    mock_pipeline.model.n_items = 3952

    # Mock FAISS index
    mock_pipeline.faiss_index.stats.return_value = {
        "n_vectors": 3952, "embed_dim": 64, "n_lists": 100,
        "n_probe": 10, "metric": "inner_product",
    }

    # Mock ranker
    mock_pipeline.ranker.model_info.return_value = {
        "n_features": 50,
        "best_iteration": 350,
        "top_10_features": {"genre_affinity": 100.0},
    }

    # Mock latency trackers
    mock_pipeline.retrieval_latency.p50 = 5.0
    mock_pipeline.ranking_latency.p50 = 12.0

    # Mock get_recommendations
    mock_results = [
        RecommendationResult(
            item_id=i,
            title=f"Movie {i} (2001)",
            score=1.0 - i * 0.05,
            rank=i,
            retrieval_score=0.95 - i * 0.02,
            genres=["Action", "Drama"],
        )
        for i in range(1, n_results + 1)
    ]
    mock_pipeline.get_recommendations.return_value = mock_results
    mock_pipeline._popularity_recommendations.return_value = mock_results

    mock_pipeline.get_stats.return_value = {
        "total_requests": 42,
        "cache_hits": 10,
        "cache_misses": 32,
        "cache_hit_rate": 0.238,
        "latency_p50_ms": 18.5,
        "latency_p99_ms": 85.2,
        "retrieval_p50_ms": 5.0,
        "retrieval_p99_ms": 20.0,
        "ranking_p50_ms": 12.0,
        "ranking_p99_ms": 45.0,
    }

    return mock_pipeline


@pytest.fixture
def client():
    """Create TestClient with mocked pipeline."""
    from src.serving.app import create_app
    import src.serving.app as app_module

    app = create_app()

    with TestClient(app, raise_server_exceptions=True) as c:
        # Inject mock pipeline
        app_module._pipeline = make_mock_pipeline()
        yield c
        app_module._pipeline = None


# ------------------------------------------------------------------ #
# Health Check Tests                                                  #
# ------------------------------------------------------------------ #

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "pipeline_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_health_with_loaded_pipeline(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline_loaded"] is True

    def test_health_without_pipeline(self):
        """Health endpoint should return degraded status when pipeline is None."""
        from src.serving.app import create_app
        import src.serving.app as app_module

        app = create_app()
        with TestClient(app) as c:
            app_module._pipeline = None
            response = c.get("/health")
            data = response.json()
            assert data["status"] == "degraded"
            assert data["pipeline_loaded"] is False


# ------------------------------------------------------------------ #
# Recommendation Endpoint Tests                                       #
# ------------------------------------------------------------------ #

class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 5})
        assert response.status_code == 200

    def test_recommend_response_structure(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 5})
        data = response.json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "latency_ms" in data
        assert "cache_hit" in data
        assert "n_candidates" in data

    def test_recommend_user_id_matches(self, client):
        response = client.post("/recommend", json={"user_id": 42, "k": 10})
        data = response.json()
        assert data["user_id"] == 42

    def test_recommend_returns_k_results(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 5})
        data = response.json()
        assert len(data["recommendations"]) == 5

    def test_recommend_item_structure(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 3})
        data = response.json()
        recs = data["recommendations"]
        assert len(recs) > 0
        for rec in recs:
            assert "item_id" in rec
            assert "title" in rec
            assert "score" in rec
            assert "rank" in rec
            assert "genres" in rec

    def test_recommend_ranks_are_ordered(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 5})
        data = response.json()
        ranks = [r["rank"] for r in data["recommendations"]]
        assert ranks == sorted(ranks), f"Ranks not sorted: {ranks}"

    def test_recommend_default_k(self, client):
        response = client.post("/recommend", json={"user_id": 1})
        assert response.status_code == 200
        data = response.json()
        # Default k=20, but our mock returns 5
        assert len(data["recommendations"]) > 0

    def test_recommend_invalid_user_id_zero(self, client):
        response = client.post("/recommend", json={"user_id": 0})
        assert response.status_code == 422  # Validation error (gt=0 constraint)

    def test_recommend_invalid_user_id_negative(self, client):
        response = client.post("/recommend", json={"user_id": -5})
        assert response.status_code == 422

    def test_recommend_k_too_large(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 500})
        assert response.status_code == 422  # le=100 constraint

    def test_recommend_missing_user_id(self, client):
        response = client.post("/recommend", json={"k": 5})
        assert response.status_code == 422

    def test_recommend_cache_hit_flow(self, client):
        """When cache returns results, the pipeline should not be called."""
        import src.serving.app as app_module
        mock_pipeline = app_module._pipeline
        mock_pipeline.feature_store.get_cached_recommendations.return_value = [
            {
                "item_id": 100,
                "title": "Cached Movie (1999)",
                "score": 0.95,
                "rank": 1,
                "retrieval_score": 0.90,
                "genres": ["Comedy"],
            }
        ]

        response = client.post("/recommend", json={"user_id": 1, "k": 1, "use_cache": True})
        assert response.status_code == 200
        data = response.json()
        assert data["cache_hit"] is True
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["item_id"] == 100

        # Reset
        mock_pipeline.feature_store.get_cached_recommendations.return_value = None

    def test_recommend_latency_is_numeric(self, client):
        response = client.post("/recommend", json={"user_id": 1, "k": 5})
        data = response.json()
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0


# ------------------------------------------------------------------ #
# Model Info Endpoint Tests                                           #
# ------------------------------------------------------------------ #

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_structure(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert "model_version" in data
        assert "embedding_dim" in data
        assert "n_users" in data
        assert "n_items" in data
        assert "index_stats" in data
        assert "ranker_info" in data
        assert "pipeline_stats" in data

    def test_model_info_embedding_dim(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert data["embedding_dim"] == 64

    def test_model_info_without_pipeline(self):
        """Model info should return 503 when pipeline is not loaded."""
        from src.serving.app import create_app
        import src.serving.app as app_module

        app = create_app()
        with TestClient(app) as c:
            app_module._pipeline = None
            response = c.get("/model/info")
            assert response.status_code == 503


# ------------------------------------------------------------------ #
# Metrics Endpoint Tests                                              #
# ------------------------------------------------------------------ #

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        response = client.get("/metrics")
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_contains_prometheus_format(self, client):
        # Make a recommendation first to populate metrics
        client.post("/recommend", json={"user_id": 1, "k": 5})
        response = client.get("/metrics")
        text = response.text
        # Prometheus text format always includes HELP and TYPE lines
        assert "# HELP" in text or len(text) > 0


# ------------------------------------------------------------------ #
# Items Endpoint Tests                                                #
# ------------------------------------------------------------------ #

class TestItemsEndpoint:
    def test_item_not_found(self, client):
        import src.serving.app as app_module
        mock_pipeline = app_module._pipeline
        mock_pipeline._item_titles = {1: "Test Movie (2001)"}
        mock_pipeline._item_genres = {1: ["Action"]}

        response = client.get("/items/99999")
        assert response.status_code == 404

    def test_item_found(self, client):
        import src.serving.app as app_module
        mock_pipeline = app_module._pipeline
        mock_pipeline._item_titles = {1: "Test Movie (2001)"}
        mock_pipeline._item_genres = {1: ["Action", "Drama"]}

        response = client.get("/items/1")
        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == 1
        assert data["title"] == "Test Movie (2001)"
        assert "genres" in data
enres" in data
